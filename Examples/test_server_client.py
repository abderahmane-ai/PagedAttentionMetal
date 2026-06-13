#!/usr/bin/env python3
import argparse
import json
import time
import sys
import urllib.request
import urllib.error

def main():
    parser = argparse.ArgumentParser(description="PagedAttention HTTP Server Python Client Demo")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--prompt", default="In the beginning, the universe was created.", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--api-key", default=None, help="Bearer authorization API Key")
    parser.add_argument("--stream", action="store_true", help="Enable SSE response streaming")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/completions"
    data = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "temperature": 0.8,
        "top_p": 0.95
    }
    encoded_data = json.dumps(data).encode("utf-8")

    headers = {
        "Content-Type": "application/json"
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    req = urllib.request.Request(url, data=encoded_data, headers=headers, method="POST")

    print(f"Sending completions request to: {url}")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Streaming: {args.stream}")
    if args.api_key:
        print("Auth: Bearer token configured")
    print("-" * 50)

    start_time = time.time()
    try:
        if args.stream:
            # Query server with streaming handler
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    token_count = 0
                    sys.stdout.write("Output: ")
                    sys.stdout.flush()
                    buffer = ""
                    while True:
                        chunk = response.read(256)
                        if not chunk:
                            break
                        buffer += chunk.decode("utf-8")
                        while "\n\n" in buffer:
                            line, buffer = buffer.split("\n\n", 1)
                            if line.startswith("data: "):
                                payload_str = line[6:]
                                if payload_str == "[DONE]":
                                    break
                                try:
                                    payload = json.loads(payload_str)
                                    if "choices" in payload:
                                        text = payload["choices"][0]["text"]
                                        token_count = payload["usage"]["completion_tokens"]
                                        sys.stdout.write(text)
                                        sys.stdout.flush()
                                except Exception as e:
                                    pass
                    print()
                    elapsed = time.time() - start_time
                    tok_s = token_count / elapsed if elapsed > 0 else 0
                    print("-" * 50)
                    print(f"Completed in {elapsed:.3f} seconds ({tok_s:.1f} tokens/sec, generated {token_count} tokens)")
                else:
                    print(f"Error status code: {response.status}")
        else:
            with urllib.request.urlopen(req) as response:
                content = response.read().decode("utf-8")
                payload = json.loads(content)
                elapsed = time.time() - start_time
                if "choices" in payload:
                    text = payload["choices"][0]["text"]
                    token_count = payload["usage"]["completion_tokens"]
                    tok_s = token_count / elapsed if elapsed > 0 else 0
                    print(f"Output: {text}")
                    print("-" * 50)
                    print(f"Completed in {elapsed:.3f} seconds ({tok_s:.1f} tokens/sec, generated {token_count} tokens)")
                else:
                    print(f"Unexpected response format: {content}")

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        error_body = e.read().decode("utf-8")
        print(f"Error Payload: {error_body}")
        if e.code == 401:
            print("Tip: Provide the correct API key using --api-key.")
        elif e.code == 429:
            print("Tip: You have exceeded the rate limits. Wait a minute before retrying.")
    except urllib.error.URLError as e:
        print(f"URL Error: Failed to connect to server: {e.reason}")

if __name__ == "__main__":
    main()
