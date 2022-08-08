from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.models import load_model

# include_top=True，表示會載入完整的 VGG16 模型，包括加在最後3層的卷積層
# include_top=False，表示會載入 VGG16 的模型，不包括加在最後3層的卷積層，通常是取得 Features
# 若下載失敗，請先刪除 c:\<使用者>\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5
#model = VGG16(weights='imagenet', include_top=True)
model = load_model("crosswalk2-model-base-VGG16.h5")
#print(model.summary())
#print(model.summary())
# Input：要辨識的影像

a = model.get_weights()
b = np.around(a[23],decimals=2)




img_path = r'C:\Users\peter\Desktop\test\08.jpg'
#img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)




# 預測，取得pred，維度為 (1,7,7,512)
features = model.predict(x)
crossing_pred = round(features[0][0],5)
other_pred = features[0][1]

print(crossing_pred)
print(other_pred)
print(crossing_pred)
print(round(other_pred,5))
#print(np.argmax(pred[0]))
#if (crossing_pred - other_pred) > 10:
#    print('正確')
#else:
#    print('不正確')
#print('Predicted:', decode_predictions(features, top=3))
#print(np.argmax(features[0]))
#print('Predicted:', decode_predictions(features, top=3)[0])
#print(model.summary())
#print('Predicted:')
