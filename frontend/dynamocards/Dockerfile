FROM node:16-alpine

WORKDIR /app


COPY . /app


EXPOSE 5173
RUN npm install

RUN npm run build
CMD ["npm", "run", "dev"]
