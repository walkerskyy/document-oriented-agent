# document-oriented-agent

Using Vector DB, Langchain and ChatGPT to get enhanced Responses and Cited Sources from Siteminder 'Help and learning' Document Database

### start application:

- copy `.env.example` to `.env` and add your OpenAI API key to .env file
- run `docker-compose up` to start container
- app will start on `localhost:1010`


OR:
- use python 3.9
- `pip install --no-cache-dir -r requirements.txt`
- `uvicorn main:app --host 0.0.0.0 --port 1010 --reload --log-level debug --use-colors`

### 

- Usage:
`GET http://0.0.0.0:1010/query`
- Payload:

```
{
	"text": "<your question here>",
	"language": "<language>"
}
```
