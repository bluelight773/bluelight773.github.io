# GPT2 Text Generation

The `transformers` library allows you to easily generate text using a GPT2 model.

1. Installation
~~~bash
pip install transformers
~~~

2. Code
~~~python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Once upon a time,", min_length=500, max_length=1000, num_return_sequences=5)
~~~

Sample text:
~~~
Once upon a time, I didn't want to work with him."

The only person who would ever find a way to bring the whole project to a close was Mark Millar. But the two met on a train, and she worked out they would build a house on the property.

But Millar had his sights set on a different sort of buyer. He wanted a home with an active public park, and while he and Mills would talk on the phone for a month, he was just as optimistic about Millar as he was anyone.

"I believe in doing something to change the world, and I'm excited I'm getting paid to do that," Millar said.

Mills was impressed with the work he's done on the project, working together through the public consultation work. Then he got in touch with Millar and asked him to start his own business.

When it came down to it, Millar said he's not planning on doing anything but work on the public park in the future, but just taking a step back and taking a chance. Millar said that he's got a passion for this thing that he's actually been building for the past year.

With the land he was looking at, though, he decided to move his store in. If you like to see the land you can try out the video on the home page.

And for now, he's doing just that. He's hoping to start one in a few weeks.

Mark Millar is on Facebook . It's a great little place to find the news.

Read more from Mark Millar's archive, follow him on Twitter or subscribe to his updates on Facebook.

His book is "I Walk the Road: A History of Man, The Future, And The Culture That Led Us In The 20th Century – a collection of essays, stories, photographs and short stories." He still has not finished it.

He's going to sell lots of his books on The Culture, but most of his book will stay on sale. If you purchase The Culture's third-quarter results, you get everything that's been posted on Amazon's Kindle store before it ends April 15.

When it comes to the project, you can still donate to his non-profit or help him get some of these money back. Check his donate link. And if you donate to the book with a friend, you can give it to Millar at my website (hulkthedoc.com) if you like.

Donate now, now!

"I love when my books are going to the right places to get the right ideas. For something that isn't a story, I just don't want to be a marketing guy," Millar said. "I want to make it as fun as possible."
~~~
