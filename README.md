# Adobe-Hackathon-Round-1b

### 1. Build the image command:

docker build -t adobe-hackathon-solution .



### 2.Run the container command:

docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none adobe-hackathon-solution
