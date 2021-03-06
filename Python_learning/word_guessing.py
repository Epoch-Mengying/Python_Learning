alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZs "

def guesser(prev_txt, rls):
    all_guesses = []
    for (suffix, guesses) in rls:
        if (suffix == None) or ((len(prev_txt) >= len(suffix)) and (prev_txt[-len(suffix):] == suffix)):
            # append all the guesses that are new
            for guess in guesses:
                if guess not in all_guesses:
                    all_guesses.append(guess)
    return all_guesses


def performance(txt, rls):
    # txt is a string
    # rls is a list of tuples; each tuple represents one rule
    # Run the guesser on txt and print out the overall performance

    tot = 0 # initialize accumulator for total guesses required
    prev_txt = ""
    for c in txt:
        to_try = guesser(prev_txt, rls)
        # find out position of the next character of txt, in the guesses list to_try
        # That's how many guesses it would take before you make the right guess
        guess_count = to_try.index(c)
        tot += guess_count
        # c has now been revealed, so add it to prev_txt
        prev_txt += c
    # done with the for loop; print the overall performance
    print("%d characters to guess\t%d guesses\t%.2f guesses per character, on average\n" % (len(txt) -1, tot, float(tot)/(len(txt) -1)))


def collapse(txt):
    # turn newlines and tabs into spaces and collapse multiple spaces to just one
    # gets rid of anything that's not in our predefined alphabet.
    txt = txt.upper()
    txt = txt.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    res = ""
    prev = ""
    for c in txt:
        if c in alphabet:
            # if not second space in a row, use it
            if c != " " or prev != " ":
                res += c
            # current character will be prev on the next iteration
            prev = c
    return res


train_txt = collapse("""
IN the year 1878 I took my degree of Doctor of Medicine of the
University of London, and proceeded to Netley to go through the course
prescribed for surgeons in the army. Having completed my studies there,
I was duly attached to the Fifth Northumberland Fusiliers as Assistant
Surgeon. The regiment was stationed in India at the time, and before
I could join it, the second Afghan war had broken out. On landing at
Bombay, I learned that my corps had advanced through the passes, and
was already deep in the enemy's country. I followed, however, with many
other officers who were in the same situation as myself, and succeeded
in reaching Candahar in safety, where I found my regiment, and at once
entered upon my new duties.

The campaign brought honours and promotion to many, but for me it had
nothing but misfortune and disaster. I was removed from my brigade and
attached to the Berkshires, with whom I served at the fatal battle of
Maiwand. There I was struck on the shoulder by a Jezail bullet, which
shattered the bone and grazed the subclavian artery. I should have
fallen into the hands of the murderous Ghazis had it not been for the
devotion and courage shown by Murray, my orderly, who threw me across a
pack-horse, and succeeded in bringing me safely to the British lines.

Worn with pain, and weak from the prolonged hardships which I had
undergone, I was removed, with a great train of wounded sufferers, to
the base hospital at Peshawar. Here I rallied, and had already improved
so far as to be able to walk about the wards, and even to bask a little
upon the verandah, when I was struck down by enteric fever, that curse
of our Indian possessions. For months my life was despaired of, and
when at last I came to myself and became convalescent, I was so weak and
emaciated that a medical board determined that not a day should be lost
in sending me back to England. I was dispatched, accordingly, in the
troopship "Orontes," and landed a month later on Portsmouth jetty, with
my health irretrievably ruined, but with permission from a paternal
government to spend the next nine months in attempting to improve it.

I had neither kith nor kin in England, and was therefore as free as
air--or as free as an income of eleven shillings and sixpence a day will
permit a man to be. Under such circumstances, I naturally gravitated to
London, that great cesspool into which all the loungers and idlers of
the Empire are irresistibly drained. There I stayed for some time at
a private hotel in the Strand, leading a comfortless, meaningless
existence, and spending such money as I had, considerably more freely
than I ought. So alarming did the state of my finances become, that
I soon realized that I must either leave the metropolis and rusticate
somewhere in the country, or that I must make a complete alteration in
my style of living. Choosing the latter alternative, I began by making
up my mind to leave the hotel, and to take up my quarters in some less
pretentious and less expensive domicile.

On the very day that I had come to this conclusion, I was standing at
the Criterion Bar, when some one tapped me on the shoulder, and turning
round I recognized young Stamford, who had been a dresser under me at
Barts. The sight of a friendly face in the great wilderness of London is
a pleasant thing indeed to a lonely man. In old days Stamford had never
been a particular crony of mine, but now I hailed him with enthusiasm,
and he, in his turn, appeared to be delighted to see me. In the
exuberance of my joy, I asked him to lunch with me at the Holborn, and
we started off together in a hansom.
""")

test_txt = collapse("""
    "Ah," said Holmes, "I think that what you have been good enough
to tell us makes the matter fairly clear, and that I can deduce
all that remains. Mr. Rucastle then, I presume, took to this
system of imprisonment?"

"Yes, sir."

"And brought Miss Hunter down from London in order to get rid of
the disagreeable persistence of Mr. Fowler."

"That was it, sir."

"But Mr. Fowler being a persevering man, as a good seaman should
be, blockaded the house, and having met you succeeded by certain
arguments, metallic or otherwise, in convincing you that your
interests were the same as his."

"Mr. Fowler was a very kind-spoken, free-handed gentleman," said
Mrs. Toller serenely.

"And in this way he managed that your good man should have no
want of drink, and that a ladder should be ready at the moment
when your master had gone out."

"You have it, sir, just as it happened."

"I am sure we owe you an apology, Mrs. Toller," said Holmes, "for
you have certainly cleared up everything which puzzled us. And
here comes the country surgeon and Mrs. Rucastle, so I think,
Watson, that we had best escort Miss Hunter back to Winchester,
as it seems to me that our locus standi now is rather a
questionable one."

And thus was solved the mystery of the sinister house with the
copper beeches in front of the door. Mr. Rucastle survived, but
was always a broken man, kept alive solely through the care of
his devoted wife. They still live with their old servants, who
probably know so much of Rucastle's past life that he finds it
difficult to part from them. Mr. Fowler and Miss Rucastle were
married, by special license, in Southampton the day after their
flight, and he is now the holder of a government appointment in
the island of Mauritius. As to Miss Violet Hunter, my friend
Holmes, rather to my disappointment, manifested no further
interest in her when once she had ceased to be the centre of one
of his problems, and she is now the head of a private school at
Walsall, where I believe that she has met with considerable success.
""")

# produce a dictionary with letters and counts
def letter_frequencies(txt):
    d = {}
    for c in txt:
        if c not in d:
            d[c] = 1
        else:
            d[c] = d[c] + 1
    return d

fs = letter_frequencies(train_txt)


# sort the letters by how frquently they appear
sorted_lets = sorted(fs.keys(), key = lambda c: fs[c], reverse=True)

def next_letter_frequencies(txt):
    # txt is a big text string
    r = {} # initialize the accumulator, an empty ditionary
    for i in range(len(txt)-1):
        # loop through the positions (indexes) of txt;
        # each iteration, we'll be looking at the
        # letter txt[i] and the following letter, txt[i+1]
        if txt[i] not in r:
            # first time we've seen the current letter
            # make an empty dictionary for counts of what letters come next
            r[txt[i]] = {}
        next_letter_freqs = r[txt[i]]  # dictionary of counts of what letters come next after txt[i]
        next_letter = txt[i+1]  # next letter is txt[i+1]
        if next_letter not in next_letter_freqs:
            # first time seeing next_letter after txt[i+1]
            next_letter_freqs[next_letter] = 1
        else:
            next_letter_freqs[next_letter] = next_letter_freqs[next_letter] + 1
    return r

counts = next_letter_frequencies(train_txt)

import math
def entropy(txt, rls):
    guess_frequencies = {}
    prev_txt = ""
    for c in txt:
        to_try = guesser(prev_txt, rls)
        guess_count = to_try.index(c) + 1
        if guess_count in guess_frequencies:
            guess_frequencies[guess_count] += 1
        else:
            guess_frequencies[guess_count] = 1
        prev_txt += c

    print("guess_frequencies:", guess_frequencies)
    # from frequencies, compute entropy
    acc = 0.0
    for i in range(len(guess_frequencies.keys())):
        guess_count = guess_frequencies.keys()[i]
        probability = guess_frequencies[guess_count] / float(len(txt))
        if i < len(guess_frequencies.keys()) - 1:
            next_guess_count = guess_frequencies.keys()[i+1]
            next_probability = guess_frequencies[next_guess_count] / float(len(txt))
        else:
            next_probability = 0
        acc += guess_count * (probability-next_probability) * math.log(guess_count, 2)

    print("entropy:", acc)
    return acc
