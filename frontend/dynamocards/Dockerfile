FROM node:14-alpine AS frontend

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json /app/frontend
RUN npm install

COPY  ./app/frontend

WORKDIR /app/frontend/dynamocards

EXPOSE 3000

CMD["npm","run","dev"]