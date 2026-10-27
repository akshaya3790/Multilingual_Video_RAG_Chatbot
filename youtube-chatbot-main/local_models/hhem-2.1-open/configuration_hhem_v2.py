from transformers import PretrainedConfig

class HHEMv2Config(PretrainedConfig):
    model_type = "HHEMv2"
    foundation = "google/flan-t5-base"
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    
    def __init___(self,
                  foundation="xyz",
                  prompt="abc",
                  **kwargs):
        super().__init__(**kwargs)
        self.foundation = foundation
        self.prompt = prompt


# FIXME: The default values passed to the constructor are not used.
# Instead, the values set as global before the constructor are used. 
# To test, run this: 
# config = HHEMv2Config()
# print(config.foundation)
# The output will not be xyz but google/flan-t5-base.