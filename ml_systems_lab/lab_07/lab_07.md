# LLM Hosting on Raspberry Pi/Jetson Nano Assignment Solution

## Step 1: Download Gemma-2B
1. Connect to your Raspberry Pi/Jetson Nano via SSH:
```bash
ssh username@your_device_ip
```

2. Download the Gemma llamafile:
```bash
wget https://huggingface.co/Mozilla/gemma-2-2b-it-llamafile/resolve/main/gemma-2-2b-it.Q6_K.llamafile
```

3. Make the file executable:
```bash
chmod +x gemma-2-2b-it.Q6_K.llamafile
```

## Step 2: Chat with Teacher LLM
1. Start the LLM with the teacher context:
```bash
./gemma-2-2b-it.Q6_K.llamafile --chat -p "you are a teacher"
```

2. In the chat interface, type:
```
Implement a simple neural network in pytorch
```

## Step 3: Host Web Server
1. Start the server in a separate terminal session (or use screen/tmux):
```bash
./gemma-2-2b-it.Q6_K.llamafile --server --nobrowser --host 0.0.0.0
```

2. From your computer, send a request to the server:
```bash
curl http://YOUR_PI_IP:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gemma-2b-it",
     "messages": [
       {"role": "system", "content": "You are an user"},
       {"role": "user", "content": "How are you?"}
     ],
     "temperature": 0.0
   }'
```

### Performance Optimization
1. Monitor system resources using `top` or `htop`
2. If memory usage is high, consider:
   - Reducing the number of concurrent requests
   - Adjusting the model's quantization parameters
   - Closing unnecessary background processes