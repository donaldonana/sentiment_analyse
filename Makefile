CFLAGS= -g3 -c -Wall
CC= gcc
SRC = src
LIB = lib
OBJ = obj
BIN = bin
OBJECTS=$(OBJ)/utils.o
HEADERS= $(SRC)/utils.h 

all : app test app2

$(OBJ)/utils.o: $(SRC)/utils.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/utils.c  -o $(OBJ)/utils.o -lm -pthread

$(OBJ)/app.o: $(SRC)/app.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/app.c  -o $(OBJ)/app.o -lm -pthread

$(OBJ)/test.o: $(SRC)/test.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/test.c -o $(OBJ)/test.o -lm -pthread

app: ${OBJECTS} $(OBJ)/app.o $(HEADERS)
	$(CC) -o $(BIN)/app.exe ${OBJECTS} $(OBJ)/app.o -lm -pthread

app2: ${OBJECTS} $(OBJ)/app.o $(HEADERS)
	$(CC) $(SRC)/app.c -o app ${OBJECTS} -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

test: ${OBJECTS} $(OBJ)/test.o $(HEADERS)
	$(CC) -o $(BIN)/test.exe ${OBJECTS} $(OBJ)/test.o -lm -pthread

clean: 
	rm -rf $(OBJ)/*.o
	rm -rf $(BIN)/*.exe

empty: 
	del /F /Q $(OBJ)\*.o
	del /F /Q $(BIN)\*.exe

run-test: test
	./$(BIN)/test.exe

run-app: app
	./$(BIN)/app.exe