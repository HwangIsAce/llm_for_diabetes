
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model_path meta-llama/Llama-2-7b-hf \
    --lora_adapter_path dog-instruct-wrapper-7b-lora \
    --data_path my_document.json \
    --output_path my_task.json \
    --temperature 0.0 \
    --top_p 0.90 \
    --max_tokens 2048
