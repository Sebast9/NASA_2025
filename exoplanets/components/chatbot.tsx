"use client"

import React, { useState } from "react"
import { MessageCircle, X, Send, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"

type Message = {
  role: "user" | "assistant"
  text: string
}

export function ChatBot() {
  const [isOpen, setIsOpen] = useState(false)
  const [inputValue, setInputValue] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = { role: "user", text: inputValue }
    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    try {
      const res = await fetch("/api/chatbot", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputValue }),
      })

      const data = await res.json()

      const aiMessage: Message = {
        role: "assistant",
        text: data.summary || "No response",
      }

      setMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      console.error("Error sending message:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const openChat = () => {
    setIsOpen(true)

    // Si no hay mensajes, muestra el mensaje inicial
    if (messages.length === 0) {
      setMessages([
        {
          role: "assistant",
          text: "Hola, ¿en qué puedo ayudarte hoy?",
        },
      ])
    }
  }

  return (
    <>
      {/* Botón flotante */}
      {!isOpen && (
        <Button
          onClick={openChat}
          className="fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full shadow-2xl glow bg-primary hover:bg-primary/90 transition-all hover:scale-110"
          size="icon"
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      )}

      {/* Modal del chat */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 z-50 w-[380px] h-[500px] flex flex-col">
          <Card className="flex flex-col h-full border-2 border-primary/30 bg-background/95 backdrop-blur-xl shadow-2xl glow">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border/50 bg-primary/5">
              <div className="flex items-center gap-2">
                <div className="rounded-full bg-primary/20 p-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">Asistente IA</h3>
                  <p className="text-xs text-muted-foreground">Experto en Exoplanetas</p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="h-8 w-8">
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Mensajes */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-2 ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-foreground border border-border/50"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                  </div>
                </div>
              ))}

              {/* Indicador de escritura */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted border border-border/50 rounded-2xl px-4 py-2">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce [animation-delay:0.2s]" />
                      <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce [animation-delay:0.4s]" />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="p-4 border-t border-border/50 bg-muted/20">
              <div className="flex gap-2">
                <Input
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Escribe tu pregunta..."
                  disabled={isLoading}
                  className="flex-1 border-border/50 bg-background/50 focus:border-primary/50"
                />
                <Button
                  type="submit"
                  size="icon"
                  disabled={!inputValue.trim() || isLoading}
                  className="glow bg-primary hover:bg-primary/90"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </form>
          </Card>
        </div>
      )}
    </>
  )
}
