FROM python

WORKDIR /demo .

COPY . .

RUN pip install pandas
RUN pip install Flask
RUN pip install jsonify
RUN pip install requests
# RUN pip install render_template
RUN pip install scikit-learn
RUN pip install joblib

EXPOSE 5000
CMD [ "python", "app.py"]
