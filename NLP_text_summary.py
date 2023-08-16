from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)



def NLP_summarizer(rawdocs):
    input_text = "summarize: " + rawdocs

    tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
    summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
    summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    return summary, rawdocs

text = """Crowds had been flocking there all night and all morning, their vehicles jamming up the typically sleepy two-lane road that gets you in and out of these parts. Infrastructure-wise, Waimea Bay isn’t exactly meant to host one of the world’s most-storied sporting events. But every once in a while, when the waves are just right, it does. The famed Eddie Aikau Big Wave Invitational is the rarest of rare surf contests, one that defies human scheduling and relies instead on the whims of nature. It requires exceptionally specific conditions: Waves in Waimea Bay must reliably reach an awesome, gut-churning height of 40 feet minimum. Even though the contest—named in honor of the legendary Native Hawaiian lifeguard and big-wave surfer Eddie Aikau—has been going since 1985, this was only its 10th run.
The chosen few who are invited to compete will drop everything and fly in from Australia and Tahiti, Brazil and Portugal. So when word got out in January that the contest was on, the world’s best surfers began racing to Waimea."""    

print(NLP_summarizer(text))