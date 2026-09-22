"use client";

import { MessageSquarePlus, Trash2, X } from "lucide-react";
import type { Chat } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export function Sidebar({
  chats,
  activeId,
  onSelect,
  onCreate,
  onDelete,
  onClose,
}: {
  chats: Chat[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onClose?: () => void;
}) {
  return (
    <div className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2 px-3 py-3">
        <Button
          onClick={onCreate}
          variant="outline"
          className="h-10 flex-1 justify-start gap-2 bg-transparent"
        >
          <MessageSquarePlus className="size-4" />
          New chat
        </Button>
        {onClose ? (
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close sidebar"
            className="size-10 shrink-0 md:hidden"
          >
            <X className="size-4" />
          </Button>
        ) : null}
      </div>

      <ScrollArea className="flex-1 px-2">
        <div className="pb-4">
          {chats.length === 0 ? (
            <p className="px-3 py-6 text-sm text-muted-foreground">No conversations yet.</p>
          ) : (
            chats.map((chat) => {
              const active = chat.id === activeId;
              return (
                <div
                  key={chat.id}
                  className={cn(
                    "group relative mb-0.5 flex items-center rounded-lg",
                    active ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/50",
                  )}
                >
                  <button
                    onClick={() => onSelect(chat.id)}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "min-h-11 flex-1 truncate rounded-lg py-2.5 pl-3 pr-10 text-left text-sm",
                      active ? "font-medium text-sidebar-foreground" : "text-sidebar-foreground/75",
                    )}
                    title={chat.title}
                  >
                    {chat.title}
                  </button>
                  <button
                    onClick={() => onDelete(chat.id)}
                    aria-label={`Delete ${chat.title}`}
                    title="Delete chat"
                    /* 44px hit area, revealed on hover or keyboard focus. */
                    className={cn(
                      "absolute right-0 flex size-11 items-center justify-center rounded-lg",
                      "text-muted-foreground opacity-0 transition-opacity",
                      "hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100",
                    )}
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
