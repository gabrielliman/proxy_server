import time
import sqlite3
import gymnasium as gym

class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, str)

class WikiEnv(gym.Env):
    def __init__(self):
        """
        Initialize the environment using a local SQLite database.
        """
        super().__init__()
        self.page = None  # current Wikipedia page
        self.obs = None  # current observation
        self.lookup_keyword = None  # current lookup keyword
        self.lookup_list = None  # list of paragraphs containing current lookup keyword
        self.lookup_cnt = None  # current lookup index
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.num_searches = 0

        # Connect to the local SQLite database containing the Wikipedia dump
        db_path = "/mnt/scratch/scheduler/proxy_server/wikipedia_local.db" # UPDATE THIS to your actual database path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Optimize SQLite for faster reads
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA cache_size=-64000;") # 64MB cache

    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    @staticmethod
    def get_page_obs(page):
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    def search_step(self, entity):
        # Format the entity to match standard Wikipedia titles (spaces instead of underscores/plus signs)
        clean_entity = entity.replace("+", " ").replace("_", " ")
        
        old_time = time.time()
        cursor = self.conn.cursor()
        
        # Try to find an exact match first (case-insensitive)
        cursor.execute("SELECT text FROM articles WHERE title = ? COLLATE NOCASE", (clean_entity,))
        result = cursor.fetchone()
        
        self.search_time += time.time() - old_time
        self.num_searches += 1

        if not result:
            # Mismatch: Search for similar titles using a basic LIKE query (or FTS if you configured it)
            old_time = time.time()
            cursor.execute("SELECT title FROM articles WHERE title LIKE ? LIMIT 5", (f"%{clean_entity}%",))
            similar_results = cursor.fetchall()
            self.search_time += time.time() - old_time
            
            if similar_results:
                self.result_titles = [row[0] for row in similar_results]
                self.obs = f"Could not find {entity}. Similar: {self.result_titles}."
            else:
                self.obs = f"Could not find {entity}. No similar pages found."
        else:
            # Exact match found
            page_text = result[0]
            
            # Handle disambiguation pages
            if "may refer to:" in page_text:
                # Optionally trigger a fallback or recursive search here
                # self.search_step("[" + entity + "]") # Warning: Ensure this doesn't infinitely loop
                pass
            
            self.page = page_text
            self.obs = self.get_page_obs(self.page)
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        with open("wiki_db.txt", "a", encoding="utf-8") as f:
          f.write(f"Search Entity: {entity}\n")
          f.write(f"Observation / Wiki Result:\n{self.obs}\n")

    def step(self, action):
        reward = 0
        done = False
        action = action.strip()
        
        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            self.search_step(entity)
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = "Invalid action: {}".format(action)

        self.steps += 1
        return self.obs, reward, done, self._get_info()

    def get_time_info(self):
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }
    
    def close(self):
        """Ensure the database connection is closed when the environment is destroyed."""
        if hasattr(self, 'conn'):
            self.conn.close()