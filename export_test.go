package llama

// ParseChatOutput renders messages through a chat session and parses chunks of
// model output fed one at a time, the last as final. It lets tests check
// reasoning extraction without generating.
func ParseChatOutput(m *Model, messages []ChatMessage, opts ChatOptions, chunks ...string) (content, reasoningContent string, err error) {
	session, err := newChatSession(m, messages, opts, opts.ReasoningFormat)
	if err != nil {
		return "", "", err
	}
	defer session.close()

	for i, chunk := range chunks {
		if content, reasoningContent, err = session.feed(chunk, i == len(chunks)-1); err != nil {
			return "", "", err
		}
	}
	return content, reasoningContent, nil
}
