from pathlib import Path

class TokenizerBuilder:
    def __init__(self, vocab_size=40000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def build_from_file(self, file_path, output_dir="tokenizers", output_name="tokenizer.json"):
        output_path = output_dir + "/" + output_name
        paths = [file_path]
        return self.build(paths, output_path)
    
    def build_from_directory(self, data_dir, output_dir="tokenizers", output_name="tokenizer.json"):
        output_path = output_dir + "/" + output_name
        paths = [str(p) for p in Path(data_dir).rglob("*.txt")]
        if len(paths) == 0:
            raise Exception(f"No .txt files in {data_dir}")
        return self.build(paths, output_path) 
    
    def build_from_dataset(self, dataset, output_dir, output_name, text_column):
        output_path = output_dir + "/" + output_name
        return self.build(output_path=output_path, dataset=dataset, text_column=text_column)    

    def build(self, paths, output_path):
        raise NotImplementedError() 