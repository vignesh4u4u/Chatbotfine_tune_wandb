def split_into_chunks(text, chunk_size):
    text_chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        text_chunks.append(chunk)
    return text_chunks

text = "This is an example text that will be split into chunks."
chunk_size = 1000

chunks = split_into_chunks(text, chunk_size)
print(chunks)
