from flask import Flask, request
import os
import time

app = Flask(__name__)

os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


print("Cache loading.....")

# To use GPTCache, that's all you need
# -------------------------------------------------
from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import OnnxModelEvaluation, SbertCrossencoderEvaluation
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
# cache.init()
# cache.set_openai_key()
onnx = Onnx()
onnx_eval = OnnxModelEvaluation()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension), max_size=5, clean_size=1, eviction="LRU")

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=onnx_eval
    )
# -------------------------------------------------

def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

@app.route('/', methods=['POST'])
def handle_request():
    data = request.get_json()
    if data and 'text' in data:
        prompt = data['text']

        start_time = time.time()
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
            {
                'role': 'user',
                'content': prompt
            }
            ],
        )
        # print(f'Question: {prompt}')
        # print("Time consuming: {:.2f}s".format(time.time() - start_time))
        # print(f'Answer: {response_text(response)}\n')
        return response_text(response)
    return "Invalid request", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
