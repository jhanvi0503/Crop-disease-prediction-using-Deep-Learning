from tensorflow.keras.models import load_model

model = load_model("mobilenetv2_10k_ft.h5", compile=False)
model.save("mobilenetv2_fixed.keras")

print("MODEL CONVERTED SUCCESSFULLY")