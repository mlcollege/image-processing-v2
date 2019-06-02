#!/bin/bash
jupyter nbextension enable widgetsnbextension --user --py
jupyter notebook --ip='0.0.0.0' --NotebookApp.token='' --allow-root --port 8000 ./notebooks/
