# Makefile
build:
    go build -o bin/soft-crusher ./cmd/go-soft-crusher

test:
    go test ./...

run:
    go run ./cmd/go-soft-crusher

docker-build:
    docker build -t soft-crusher .

docker-run:
    docker run -p 8080:8080 soft-crushe