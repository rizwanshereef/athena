import flask
from flask_mail import Mail,Message
import pandas as pd
import tensorflow as tf
import keras
keras.__version__
from keras.models import load_model

app = flask.Flask(__name__)
mail=Mail(app)


app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT']= 465
app.config['MAIL_USERNAME']='athenamcet@gmail.com'
app.config['MAIL_PASSWORD']='athena@2019'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_SUPPRESS_SEND']= False
mail=Mail(app)

# we need to redefine our loss function in order to use it when loading the model 
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom loss function
global graph
graph = tf.get_default_graph()
model = load_model('mod.h5', custom_objects={'auc': auc})

# define a predict function as an endpoint
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/resume',methods=['GET','POST'])
def resume():
    if flask.request.method=='GET':
        return flask.render_template('resume.html')
    usr=flask.request.form.get('Name')
    m=flask.request.form.get('Email')
    p1=float(flask.request.form.get('HighSchool'))
    p2=float(flask.request.form.get('CGPA'))
    p3=float(flask.request.form.get('Experience'))
    link="http://127.0.0.1:5000/predict?"+"P1="+ str(p1)+"&P2="+str(p2)+"&P3="+str(p3)
    msg = Message('Shortlisting', sender = 'athenamcet@gmail.com', recipients = [m])
    msg.body = "Dear "+str(usr)+",\nYou have been shortlisted!!\nFor further information visit www.corporate.com\nThank You"
    if (p1>=75 and p2>=7.5 and p3<5):
        mail.send(msg)
    else:
        msg.body = "Dear "+str(usr)+",\nUnfortunately you have not been shortlisted!!\nThank You"
        mail.send(msg)    
    return flask.redirect(link)  
    
    
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, echo the msg parameter 
    if (params != None):
        x = pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(model.predict(x)[0][0])
            data["success"] = True
   
    

    # return a reponse in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run()
