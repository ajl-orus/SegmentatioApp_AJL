# Testar o TFLite 
import tensorflow as tf
import numpy as np

# Carregar modelo
interpreter = tf.lite.Interpreter(model_path='/content/tf_model/unet_float32.tflite')
interpreter.allocate_tensors()

# Verificar entrada/saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(" Input:", input_details[0]['shape'], input_details[0]['dtype'])
print(" Output:", output_details[0]['shape'], output_details[0]['dtype'])

# Testar com dados aleatórios
input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f" TFLite funcionando! Output shape: {output.shape}")
