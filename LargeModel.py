import torch
import transformers

class LargeModel:
    def __init__(self, model_name):
        """
        Initialize a large model by specifying the model name.

        Args:
            model_name (str): Name of the large model to be loaded.
        """
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def generate_text(self, input_text):
        """
        Generate text using the large model.

        Args:
            input_text (str): Input text for text generation.

        Returns:
            str: Generated text.
        """
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=100)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Example usage
model = LargeModel('gpt2')
input_text = "Once upon a time"
generated_text = model.generate_text(input_text)
print(generated_text)
