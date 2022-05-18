from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    select1=request.form.get('crop1')
    select2=request.form.get('crop2')
    input_query = np.array([[select1,select2]])
    result = model.predict(input_query)[0]
    return jsonify({'label':str(result)})
if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    