-- One message per session. save_message saves it or replaces it, so the caller can add to
-- or correct their message during the call, and it's emailed once, when the call ends.
DROP INDEX messages_session;
CREATE UNIQUE INDEX messages_session ON messages(session_id);
