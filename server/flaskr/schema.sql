DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL
);


CREATE TABLE prediction (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  key_id TEXT NOT NULL,
  input_files TEXT NOT NULL,
  results_fiels TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES user (id)
);
