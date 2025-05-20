CC = gcc
CCFLAGS = -O3 -march=native -ffast-math -Wall
LDFLAGS = -lm
APP_BINARY = nanochat

all: $(APP_BINARY)

$(APP_BINARY): main.c oled.c ui.c avltree.c bpe.c hashmap.c trie.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

llm: main_llm.c bpe.c hashmap.c trie.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

install:
	chmod +x $(APP_BINARY)
	cp -pR $(APP_BINARY) /usr/bin/
	grep -qxF "/usr/bin/nanochat &" /etc/rc.local || sed -i "/^[[:space:]]*exit[[:space:]]0/i /usr/bin/nanochat &" /etc/rc.local

clean:
	rm -f $(APP_BINARY)
	rm -f llm
