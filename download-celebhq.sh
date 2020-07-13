# Before running this script, download `https://drive.google.com/file/d/1xYnk8eU0zJLoX5iVt_FGjJxDYKj3VsN_/view?usp=sharing` and save to `data/celebhq/images.zip`.

FILE="data/celebhq/images.zip"
if test -f "$FILE"; then
    echo "Unpacking zip file."
else
    echo "Download file from https://drive.google.com/file/d/1xYnk8eU0zJLoX5iVt_FGjJxDYKj3VsN_/view?usp=sharing and save to data/celebhq/images.zip before rerunning this script."
    exit
fi

unzip data/celebhq/images.zip -d data/celebhq/
rm -f data/celebhq/images.zip
