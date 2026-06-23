# Inference Guide

> This guide is based on the original inference code provided by WeiboAI. See the [original README](../ORIGINAL_README.md) for full context.

## Recommended Parameters

| Parameter | VibeThinker-3B | VibeThinker-1.5B |
|-----------|---------------|-----------------|
| temperature | 0.6 or 1.0 | 0.6 or 1.0 |
| top_p | 0.95 | 0.95 |
| top_k | -1 (skip) | -1 (skip) |
| max_new_tokens | 40960 | 40960 |

## Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class VibeThinker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            torch_dtype="bfloat16",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

    def infer_text(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        generation_config = dict(
            max_new_tokens=40960,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=None
        )
        generated_ids = self.model.generate(
            **model_inputs,
            generation_config=GenerationConfig(**generation_config)
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids
            )
        ]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        return response


if __name__ == '__main__':
    model = VibeThinker("WeiboAI/VibeThinker-3B")
    prompt = "What is the sum of the first 100 prime numbers?"
    print(model.infer_text(prompt))
```

## Using vLLM (Recommended)

```python
from vllm import LLM, SamplingParams

model = LLM(
    "WeiboAI/VibeThinker-3B",
    dtype="bfloat16",
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=-1,
    max_tokens=40960,
)

prompt = "What is the sum of the first 100 prime numbers?"
outputs = model.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

## Using SGLang

```python
import sglang as sgl

@sgl.function
def reasoning(s, prompt):
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("answer", max_tokens=40960))

model = sgl.Engine(model_path="WeiboAI/VibeThinker-3B")
state = reasoning.run(prompt="What is 23 * 47?")
print(state["answer"])
```

## Hardware Notes

| Model | Precision | Min VRAM | Recommended GPU |
|-------|-----------|----------|----------------|
| VibeThinker-3B | bfloat16 | ~8 GB | RTX 3070+ / A10G+ |
| VibeThinker-1.5B | bfloat16 | ~4 GB | RTX 2060+ / T4+ |

For CPU inference, use `device_map="cpu"` or no device map at all (but expect significantly slower performance).
