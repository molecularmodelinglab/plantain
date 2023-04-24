from flask import Flask, render_template
import json

app = Flask(__name__, template_folder=".")

@app.route('/')
def index():
    # Render the main HTML template
    return render_template('index.html')

@app.route('/moldata')
def moldata():
    # Load a sample molecule from a file
    molfile = "/home/boris/Data/BigBindStructV2/AL7A1_HUMAN_27_537_0/4zul_un1_lig.sdf"
    with open(molfile, 'r') as f:
        moldata = f.read()

    # Return the molecule data as JSON
    return json.dumps({'moldata': moldata})

if __name__ == '__main__':
    app.run(debug=True)