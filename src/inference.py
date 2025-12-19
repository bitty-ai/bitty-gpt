# This is to generate some output generations 

import os
import torch
import torch.nn as nn 

class Inference:
    def __init__(self, model=None, tokenizer=None) -> None:
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.model_path = os.path.join(self.data_dir, 'model.pt')
        self.tokenizer_path = os.path.join(self.data_dir, "tokenizer.pt")

        if model is not None:
            self.model = model
        elif os.path.isfile(self.model_path):
            self.model = torch.load(self.model_path, map_location='cpu')
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not provided and not found at {self.model_path}")

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif os.path.isfile(self.tokenizer_path):
            # Adjust this line if your tokenizer needs custom loading
            self.tokenizer = torch.load(self.tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer not provided and not found at {self.tokenizer_path}")

    def generate_sample(self, prompt="Once upon a time ", max_new_tokens=200, temperature=1.0, top_k=20, device="cpu", stop_token = 0):
        # Encode the prompt
        if hasattr(self.tokenizer, 'encoder'):
            input_ids = self.tokenizer.encoder(prompt)

        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.unsqueeze(0).to(device)  # (1, prompt_length)

        self.model.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(generated)  # (1, T, vocab_size)
                next_token_logits = outputs[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    probs = torch.full_like(next_token_logits, float('-inf'))
                    probs.scatter_(1, indices, values)
                    next_token_logits = probs

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            if next_token == stop_token:
                break # if its <eos>

        # Decode the output
        output_tokens = generated.squeeze(0).tolist()
        if hasattr(self.tokenizer, 'decoder'):
            decoded_text = self.tokenizer.decoder(output_tokens)

        print(f"Prompt: {prompt}")
        print(f"Generated: {decoded_text}")
        return decoded_text

    def save(self):
        # save 
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(self.tokenizer, self.tokenizer_path)
        
