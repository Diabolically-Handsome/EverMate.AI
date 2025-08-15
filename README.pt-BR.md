# 🐾 EverMate.AI (v1.0 em breve)
Seu amigo IA local · Privacidade em primeiro lugar · Offline sempre que possível

[English (Canada)](README.en-CA.md) · [中文](README.zh-CN.md) · [Português (Brasil)](README.pt-BR.md) · [Français (Canada)](README.fr-CA.md) · [日本語](README.ja-JP.md)

## ✨ O que é
O EverMate.AI torna **conversas de companhia de longo prazo** práticas e sob seu controle: roda localmente, não envia dados por padrão e organiza tudo com dois diferenciais — **Memória em Três Camadas** + **Índice Local Escalável**.

## 🍱 Nossos diferenciais
### 1) Memória em Três Camadas (Core → Persona → Vault)
- **Core**: tópicos frequentes + pistas de estilo/resposta (ex.: tratar por “você”, trazer exemplos, tom acolhedor). A maioria das perguntas acerta aqui primeiro — mais rápido e consistente.  
- **Persona**: preferências de comunicação, densidade de informação, interesses de longo prazo e “não me diga isso” — em até 8 bullets. Usa LLM local via Ollama quando disponível; caso contrário, usa heurísticas.  
- **Vault**: o restante fica **fragmentado em arquivos no disco** e é recuperado sob demanda — nada de textão infinito, só o trecho certo na hora certa.  
- **Conflitos**: se o histórico divergir do que você acabou de dizer, **vale o que acabou de dizer**.

### 2) Índice Local Escalável (Grande · Local · Rápido)
- **Fragmentação**: ~2.8K caracteres por bloco (ajustável), feito para milhões de palavras.  
- **Índice invertido**: SQLite com `terms/postings/chunks`, usando WAL para leitura/gravação estáveis.  
- **BM25**: ranqueia blocos e extrai **a melhor frase original** como evidência para o contexto.  
- **Construção contínua & atualizações incrementais**: arraste `.docx/.txt` e indexamos enquanto lemos; em conversas novas, cada turno é adicionado e Core/Persona se atualizam periodicamente.

## 🚀 Guia rápido
1. Rode `python app.py`.  
2. Dois caminhos:  
   - **Importar histórico**: arraste `.docx/.txt`, clique em “Construir/Re-construir Memória” e comece a conversar.  
   - **Novo amigo**: só conversar. Cada turno entra no índice; Core/Persona se atualizam após certos marcos.  
3. Opcional: configure **Ollama** (`OLLAMA_URL`, `OLLAMA_MODEL`) para refinar a Persona localmente.

## 💼 Como funciona (em uma linha)
Começa no **Core** (estilo e temas), passa pela **Persona** (seu jeito), e puxa o **Vault** para citações. Contexto **curto e certeiro**.

## 🧲 Importar vs. Criar novo
- **Arrastar & soltar**: `.docx` via biblioteca padrão; `.txt` em streaming.  
- **Novo amigo**: `append_turn` grava incrementalmente; atualização padrão a cada **20** novos blocos.

## 🔧 Ajustes
`CHUNK_CHARS`=2800 · `CORE_TOP_TERMS`=50 · `PERSONA_MAX_BULLETS`=8 · `REFRESH_EVERY`=20 · `retrieve(query,k)`=4–8

## 🛡️ Privacidade e local-first
Tudo roda e fica no seu computador. Se usar modelo remoto, revise suas políticas. Recomendamos criptografar e fazer backup da pasta `memory`.

## ❓FAQ
- A Persona não mudou? Provavelmente abaixo do limiar de atualização — continue conversando ou reconstrua; pode reduzir `REFRESH_EVERY`.  
- Dá para ver a **frase original**? Sim — o Vault retorna a sentença/trecho mais relevante.  
- O índice está crescendo demais? Aumente `CHUNK_CHARS`, arquive `chunks` antigos, ou separe a `memory` por amigo.  
- Sem LLM local? Só a qualidade da Persona cai; o fluxo segue com heurísticas.  
- Parece “preso” ao passado? O que você diz **agora** manda; ajuste `01_core.md` / `02_persona.md` e reconstrua.

## 🗺️ Roadmap
Retrieval híbrido (BM25 + embeddings locais), tópicos/linha do tempo, painel explicável, decaimento de memória, cartões de eventos.

## 📦 Estrutura
```
memory/
  index.sqlite
  chunks/
  uploads/
  buffer.txt
  01_core.md
  02_persona.md
  03_vault.md
```

## 📜 Licença
MIT (veja LICENSE)
