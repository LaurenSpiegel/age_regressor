var el = x => document.getElementById(x);

function showPicker(inputId) { el('file-input').click(); }

function showPicked(input) {
    // el('upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('image-picked').src = e.target.result;
        el('image-picked').className = '';
        el('image-picked').style.filter = '';
    }
    reader.readAsDataURL(input.files[0]);
    el('original').style.display = 'none'
}

function analyze() {
    var uploadFiles = el('file-input').files;
    if (uploadFiles.length != 1) {
        alert('Please select 1 file to analyze!');
        return;
    }

    el('analyze-button').innerHTML = 'Youthifying...';
    var xhr = new XMLHttpRequest();
    var loc = window.location
    xhr.open('POST', `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`, true);
    xhr.onerror = function() {alert (xhr.responseText);}
    xhr.onload = function(e) {
        if (this.readyState === 4) {
            var response = JSON.parse(e.target.responseText);
            el('image-picked').style.filter = 'blur(2px)';
            el('result-label').innerHTML = `You look ${response['result']} and <b><i>OH SO FINE</i></b>!!!`;
        }
        el('analyze-button').innerHTML = 'Get 10 Years Younger';
    }

    var fileData = new FormData();
    fileData.append('file', uploadFiles[0]);
    xhr.send(fileData);
}

