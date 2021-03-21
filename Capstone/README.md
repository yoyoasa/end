# Capstone Project


### Data Cleaning

First and the formost process in any machine learning problem is to look into data, the dataset consista of questions and their equivalent codes

#### These were following issues with data set  

1. Some questons started with # and others started with numbers so we had to make them all start with a question like syntax like `write a program`
2. Also to identify that if a perticular statement is a question or a comment we had to write rules to clean comments up first
3. Indentation problem, some programs were having tabs some had spaces and the other had mixed characters like tab, spaces

Following are some of the rules used for cleaning up dataset
    `Manually Preprocess Data ie remove any extra '# ' from data set`
    ` def` -> `def`
    `#write` -> `# write`
    `\n#\s?\d+` -> `\n# `  // Convert all numbered statements to un numbered
    `\d+\.\s?\n# write` -> `# write`
    `# Define` -> `# write`
    `\n#.python ? 3? ?` -> `\n# write `
    `\n#.Write (?!a)` -> `\n# write a`
    `\n#.Write` -> `\n# write`
    `#.program` -> `# write a program`
    `\n\s*\n\s*\n+` -> `\n\n`,
    `\n?# In\[\d+\]:\s*\n?` -> `\n`
    `\n\s*\n(?!#.write)` -> `\n\n`
    `Remove all un necessary comments`
    `Remove all un necessary new lines`
    `Remove all extra spcaes and replace them with tabs`
    `Split each program into pairs of statement and code`
    `(#.program|#.write)`

### Tokenization

As ths is a unique dataset where there are a couple number of special characters and other keywords which needs to be together

#### Following were some of the considerations needed to take in picture while writing a tokenizer
1. names can have `_` (underscore)
2. there are characters like `\n` and `\t` they needs to be a single word

To cater to these needs we had to write out custom tokenizer extending `spacy` tokenizer

    ```
    from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
    from spacy.tokenizer import Tokenizer

    def custom_tokenizer(nlp):
        infix_re = re.compile(
            r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'\(\)\[\]\{\}\*\%\^\+\-\=\<\>\|\!(//)(\n)(\t)~]''')
        prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                        suffix_search=suffix_re.search,
                        infix_finditer=infix_re.finditer,
                        token_match=None)


    spacy_que = spacy.load('en_core_web_sm')
    spacy_ans = spacy.load('en_core_web_sm')
    spacy_ans.tokenizer = custom_tokenizer(spacy_ans)


    def tokenize_que(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_que.tokenizer(text)]


    def tokenize_ans(text):
        """
        Tokenizes Code text from a string into a list of strings
        """
        return [tok.text for tok in spacy_ans.tokenizer(text)]
    ```

### Pretrained Embeddings

It is already proven that using pretrained embedding makes network learn faster and converge easier with a less number ot epochs

#### We trained a custom glove embedding with 300 dimensions 

we trained this glove embedding for 100 epochs and reached a descent loss (9.974439516472449e-13)

[Loss Graph](data:image/png;base64,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)