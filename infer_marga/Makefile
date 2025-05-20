CC = gcc
CCFLAGS = -O3 -march=native -ffast-math -Wall
LDFLAGS = -lm

all: marga cli wss

marga: nano_marga

nano_marga: main_marga.c oled.c ui.c avltree.c bpe.c hashmap.c trie.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

cli: nano_cli

nano_cli: main_cli.c bpe.c hashmap.c trie.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

wss: nano_wss

nano_wss: main_wss.c bpe.c hashmap.c trie.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lwebsockets

install:
	chmod +x nano_wss
	chmod +x nano_cli
	chmod +x nano_marga
	cp -pR nano_marga /usr/bin/
	grep -qxF "/usr/bin/nano_marga &" /etc/rc.local || sed -i "/^[[:space:]]*exit[[:space:]]0/i /usr/bin/nano_marga &" /etc/rc.local

clean:
	rm -f nano_marga
	rm -f nano_cli
	rm -f nano_wss
