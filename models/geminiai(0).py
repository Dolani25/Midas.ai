import google.generativeai as genai

GOOGLE_API_KEY = 'AIzaSyDyP6OcYXvCjzLfDQt1cfcp6G_rTfttFdU'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('write a easy to understand tutorial on how to use django for absolute beginners, for a example a site thats an AI that when given PDFs ,it will read the PDFs then teach the information gained to students in interesting informative fun bits ,so the lesson will not be boring . the target audience are students, so they will just take their textbook in pdf form, upload it to the Ai, the AI will go through the textbook and explain to them better than in a playful informative fun way .for specific features I want the app to work so you can upload your course outline so AI, the AI will go through your topics while referencing the uploaded textbook and sometimes crawling the web for more info about that topic,  it will draft a teaching plan or lesson plan  .It  will give the students practice question and after then students Can chat with the AI Iike theyre are chatting with a real person,the student  will summarise what he/she has learnt ,the AI will take notes of things the students understand very well and those the students dont understand,the AI will re-teach it in a simpler form')

print(response.text)