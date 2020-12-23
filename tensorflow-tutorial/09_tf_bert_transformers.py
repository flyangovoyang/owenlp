import tensorflow as tf
from transformers import TFBertModel, BertConfig, BertTokenizerFast
# transformers==3.0.2

print(tf.__version__)
config = BertConfig.from_pretrained('../../bert-base-chinese/config.json')
model = TFBertModel.from_pretrained('tf-model.h5', config=config)
model.summary()

tokenizer = BertTokenizerFast('../../bert-base-chinese/vocab.txt')
max_seq_len = 50


texts = ["我喜欢你", "我爱你", "我讨厌你"]
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)


for text in texts:
    be = tokenizer.encode_plus(text, truncation=True, max_length=max_seq_len+2, padding='max_length', return_tensors="tf")
    with tf.GradientTape() as tape:
        last_hidden_states, pooler_output = model(be)[:2]
        loss = loss_object(pooler_output)
    print(last_hidden_states.shape)
    print(pooler_output.shape)

# model.save('bert_save')
