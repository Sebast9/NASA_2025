// app/api/Gemini/route.ts

import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { convertToModelMessages, streamText, type UIMessage } from 'ai';

const google = createGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY!,
});

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();
  const prompt = convertToModelMessages(messages);

  const result = await streamText({
    model: google.chat('gemini-1.5-flash'),
    prompt,
    system: `Eres un asistente experto en exoplanetas...`,
    abortSignal: req.signal,
  });

  return result.toUIMessageStreamResponse();
}
