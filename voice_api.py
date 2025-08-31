import os
import sys
import json
import base64
import asyncio
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
from modules.commons import *
from hf_utils import load_custom_model_from_hf
import yaml
import soundfile as sf
import io
from argparse import Namespace
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaStreamTrack, MediaRecorder
import aiohttp
import json
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_set = None
fp16 = False
pcs = set()

# Global variables for custom_infer function
prompt_condition = None
mel2 = None
style2 = None
reference_wav_name = ""
prompt_len = 3.0
ce_dit_difference = 2.0

class SessionState:
    def __init__(self):
        self.reference_wav = None
        self.reference_wav_name = ""
        self.prompt_condition = None
        self.mel2 = None
        self.style2 = None
        self.prompt_len = 3.0
        self.ce_dit_difference = 2.0
        self.extra_frame = 0
        self.extra_frame_right = 0
        self.return_length = 0
        self.zc = 44100 // 50  # Sample rate dependent

class WebRTCVoiceTrack(MediaStreamTrack):
    def __init__(self, session_state):
        super().__init__()
        self.session_state = session_state
        self.kind = "audio"
        self._queue = asyncio.Queue()

    async def recv(self):
        frame = await self._queue.get()
        return frame

    async def process_audio(self, audio_data):
        # Process audio data through voice conversion
        output = await run_vc_inference(
            torch.from_numpy(audio_data).to(device),
            self.session_state,
            diffusion_steps=10,
            inference_cfg_rate=0.7
        )
        if output is not None:
            await self._queue.put(output.cpu().numpy().tobytes())

class WebRTCManager:
    def __init__(self):
        self.pcs = set()

    async def create_peer_connection(self, websocket: WebSocket, session_state: SessionState):
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                voice_track = WebRTCVoiceTrack(session_state)
                pc.addTrack(voice_track)
                asyncio.create_task(self.handle_audio_track(track, voice_track))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        return pc

    async def handle_audio_track(self, track, voice_track):
        while True:
            try:
                frame = await track.recv()
                audio_data = frame.to_ndarray()
                await voice_track.process_audio(audio_data)
            except Exception as e:
                print(f"Error processing audio: {e}")
                break

webrtc_manager = WebRTCManager()

class WebRTCOffer(BaseModel):
    sdp: str
    type: str

class WebRTCIceCandidate(BaseModel):
    candidate: str
    sdpMid: str
    sdpMLineIndex: int

@app.websocket("/ws/webrtc")
async def websocket_webrtc(websocket: WebSocket):
    await websocket.accept()
    session = SessionState()
    pc = await webrtc_manager.create_peer_connection(websocket, session)

    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "offer":
                # Handle WebRTC offer
                offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                await pc.setRemoteDescription(offer)
                
                # Create and send answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                })
            
            elif data["type"] == "candidate":
                # Handle ICE candidate
                candidate = RTCIceCandidate(
                    candidate=data["candidate"],
                    sdpMid=data["sdpMid"],
                    sdpMLineIndex=data["sdpMLineIndex"]
                )
                await pc.addIceCandidate(candidate)
            
            elif data["type"] == "config":
                await handle_config(websocket, data, session)
                
    except WebSocketDisconnect:
        print("Client disconnected")
        await pc.close()
        webrtc_manager.pcs.discard(pc)
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close(code=1011)

@torch.no_grad()
def custom_infer(model_set,
                reference_wav,
                new_reference_wav_name,
                input_wav_res,
                block_frame_16k,
                skip_head,
                skip_tail,
                return_length,
                diffusion_steps,
                inference_cfg_rate,
                max_prompt_length,
                cd_difference=2.0):
    """Voice conversion inference function"""
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference
    
    # Initialize ce_dit_difference if not set
    if 'ce_dit_difference' not in globals():
        global ce_dit_difference
        ce_dit_difference = 2.0
    
    (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    ) = model_set
    
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]
    
    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
    
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).float().to(device)

        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor.float(), sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0).float(), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        mel2 = to_mel(reference_wav_tensor.unsqueeze(0).float())
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    converted_waves_16k = input_wav_res
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0).float())
    
    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
    
    cond = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=None
    )[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target).squeeze()
    
    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len: -tail_len]

    return output

async def load_models(args):
    """Load all required models"""
    global model_set, fp16
    fp16 = args.get('fp16', False)
    
    if args.get('checkpoint_path') is None:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_uvit_tat_xlsr_ema.pth",
            "config_dit_mel_seed_uvit_xlsr_tiny.yml"
        )
    else:
        dit_checkpoint_path = args['checkpoint_path']
        dit_config_path = args['config_path']
    
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                        load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        vocoder_fn = vocos.decoder

    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import Wav2Vec2FeatureExtractor, HubertModel
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    
    
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    model_set = (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    args = {
        'checkpoint_path': None,
        'config_path': None,
        'fp16': True,
        'gpu': 0
    }
    await load_models(args)

@app.websocket("/ws/vc")
async def websocket_vc(websocket: WebSocket):
    """WebSocket endpoint for real-time voice conversion"""
    await websocket.accept()
    session = SessionState()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'config':
                await handle_config(websocket, data, session)
            elif data['type'] == 'audio':
                await handle_audio(websocket, data, session)
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close(code=1011)

async def handle_config(websocket: WebSocket, data: dict, session: SessionState):
    """Handle configuration messages"""
    try:
        config = data['config']
        
        if 'reference_audio' in config:
            audio_bytes = base64.b64decode(config['reference_audio'])
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            session.reference_wav = audio_array
            session.reference_wav_name = config.get('reference_name', 'default')
        
        session.prompt_len = config.get('max_prompt_length', 3.0)
        session.ce_dit_difference = config.get('ce_dit_difference', 2.0)
        
        await websocket.send_json({
            'type': 'status',
            'message': 'Configuration updated',
            'status': 'success'
        })
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'message': f'Config error: {str(e)}'
        })

async def handle_audio(websocket: WebSocket, data: dict, session: SessionState):
    """Process audio chunks through voice conversion"""
    try:
        audio_data = base64.b64decode(data['audio'])
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        sr = data['sample_rate']
        
        if len(audio_array) == 0:
            await websocket.send_json({
                'type': 'error',
                'message': 'Empty audio chunk received'
            })
            return

        input_wav = torch.from_numpy(audio_array).float().to(device)
        
        # Get ce_dit_difference from request or use default
        ce_dit_difference = data.get('ce_dit_difference', 2.0)
        
        output = await run_vc_inference(
            input_wav,
            session,
            diffusion_steps=data.get('diffusion_steps', 10),
            inference_cfg_rate=data.get('inference_cfg_rate', 0.7),
            ce_dit_difference=ce_dit_difference
        )
        
        if output is not None and output.numel() > 0:
            await websocket.send_bytes(output.cpu().numpy().tobytes())
        else:
            await websocket.send_json({
                'type': 'error',
                'message': 'Empty output from model'
            })
            
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'message': f'Audio processing error: {str(e)}'
        })

async def run_vc_inference(input_wav: torch.Tensor, 
                          session: SessionState,
                          diffusion_steps: int = 10,
                          inference_cfg_rate: float = 0.7,
                          ce_dit_difference: float = 2.0) -> Optional[torch.Tensor]:
    """Run voice conversion inference"""
    global model_set
    
    if session.reference_wav is None:
        raise ValueError("Reference audio not set")
    
    sr = model_set[-1]["sampling_rate"]
    input_wav_16k = torchaudio.functional.resample(input_wav.float(), sr, 16000)
    
    block_frame_16k = 320
    skip_head = int(ce_dit_difference * 50)
    skip_tail = skip_head
    return_length = input_wav_16k.shape[0] // 320 - skip_head - skip_tail
    
    with torch.no_grad():
        output = custom_infer(
            model_set,
            session.reference_wav,
            session.reference_wav_name,
            input_wav_16k.unsqueeze(0),
            block_frame_16k=block_frame_16k,
            skip_head=skip_head,
            skip_tail=skip_tail,
            return_length=return_length,
            diffusion_steps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
            max_prompt_length=session.prompt_len,
            cd_difference=ce_dit_difference
        )
    
    return output.squeeze(0)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model_set is not None
    }
