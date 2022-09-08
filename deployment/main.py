from fastapi import FastAPI
import pickle
import uvicorn
from lr import logistic_regression

app = FastAPI()



@app.get("/")
def read_root():
    return {'response':"Hello Nice To See You"}


@app.get("/predict/{text}")
def predict(text):
    text=text.replace('+',' ')
    return {'status':obj.predict(text)}

if __name__ == "__main__":
    file=open('logreg.pkl','rb')
    obj=pickle.load(file)
    file.close()
    uvicorn.run(app,port=9000)