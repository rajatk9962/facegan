from flask import Flask, render_template,request
from main import changefeature
from main import genrandom
from main import histimage
from main import genimage
import os
app = Flask(__name__)


count = 20
fname='static/default.jpg"'
noise=0
fake_image_history=[]
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global fname
        global count 
        global noise
        global fake_image_history
        try:    
            
            gencommand= request.form['genbutton'] 
            if gencommand:                
                for filename in os.listdir('static/'):
                    if filename.startswith('image_'):  
                        os.remove('static/' + filename)
                noise,fake_image_history,fname = genrandom()
                
        except:
            feature = request.form['feature'] 
            deg = request.form['degree']
            
                
            
            
            if deg == "inc":
                count+=1
                
                for filename in os.listdir('static/'):
                    if filename.startswith('image_'):  
                        os.remove('static/' + filename)
                noise,fake_image_history,fname=genimage(count,noise,feature,fake_image_history)
            elif deg == "dec":            
                count-=1
                
                for filename in os.listdir('static/'):
                    if filename.startswith('image_'):  
                        os.remove('static/' + filename)
                fake_image_history,fname,noise=histimage(fake_image_history)
            elif deg == "def":
                count = 20     
                for filename in os.listdir('static/'):
                    if filename.startswith('image_'):  
                        os.remove('static/' + filename)
                noise,fake_image_history,fname=changefeature(count,feature,noise)
            if not fname:
                fname = "static/default.jpg"            
 
            return render_template('index.html', feature = feature, deg = deg, count= count, image=fname)
    return render_template('index.html',image=fname)
    


if __name__ == '__main__':
    app.run()
    
