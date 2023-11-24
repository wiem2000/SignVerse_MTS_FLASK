from flask import Flask
import speech_recognition as sr

app = Flask(__name__)
r = sr.Recognizer()
result_text = ""

def speak():
    global r
    global result_text
    try:
        with sr.Microphone() as source:
            result_text="listening"
            print("Speak: ")
            audio = r.listen(source)
            text = r.recognize_google(audio)
            result_text=text
            print("YOU SAID THIS: {}".format(text))
            
    
    
         
    except :
        text="ERROR"
        result_text=text
        
        

@app.route('/start_speak', methods=['GET'])
def start_speak():
    speak()
    print("listening stopped")
    return "listening stopped"
    

@app.route('/get_result', methods=['GET'])
def get_result():
    global result_text
   
    return  result_text
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
