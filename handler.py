import os
import runpod
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio
import uuid

# ── Load config from env vars ──────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-8B-Instruct")
DTYPE = os.environ.get("DTYPE", "half")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 4096))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.90))

# ── Cold start: load model ONCE ────────────────────────────
print(f"Loading model: {MODEL_NAME}")

engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    dtype=DTYPE,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    trust_remote_code=True,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
print("Model loaded and ready!")

# ── Handler: called on every request ───────────────────────
async def handler(job):
    job_input = job["input"]

    prompt        = job_input.get("prompt", "")
    max_tokens    = job_input.get("max_tokens", 512)
    temperature   = job_input.get("temperature", 0.7)
    top_p         = job_input.get("top_p", 0.9)
    stream        = job_input.get("stream", False)

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    request_id = str(uuid.uuid4())

    if stream:
        # Streaming: yield tokens one by one
        async for output in engine.generate(prompt, params, request_id):
            yield {"token": output.outputs[0].text, "done": output.finished}
    else:
        # Non-streaming: wait for full response
        final_output = None
        async for output in engine.generate(prompt, params, request_id):
            final_output = output
        yield {"output": final_output.outputs[0].text}

runpod.serverless.start({"handler": handler})