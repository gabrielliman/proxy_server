import ast
import json
import time
import gymnasium as gym
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import logging
import sqlite3


def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return p


class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, str)


class WikiEnv(gym.Env):

    def __init__(self):
        """
          Initialize the environment.
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
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Set up standard requests session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LATS_Research/1.0 (Research/Testing; contact: gabrielliman@ufmg.br)'
        })
        retries = Retry(
            total=5, 
            backoff_factor=1, 
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=150, pool_maxsize=150)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

        # ---------------------------------------------------------
        # NEW: Custom SQLite Cache Initialization
        # ---------------------------------------------------------
        scratch_dir = "/scratch/wikiCache"
        if os.path.exists(scratch_dir):
            db_path = os.path.join(scratch_dir, 'wiki_parsed_cache.db')
        else:
            db_path = 'wiki_parsed_cache.db'

        # timeout=15 helps prevent database lock errors if running multi-threaded/multi-processing
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=15.0)
        
        # PRAGMA optimizations for fast concurrent I/O
        self.conn.execute("PRAGMA journal_mode=WAL;") 
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS wiki_cache (
                entity TEXT PRIMARY KEY,
                is_match INTEGER,
                obs TEXT,
                page TEXT,
                result_titles TEXT,
                is_redirect INTEGER,
                redirect_entity TEXT
            )
        ''')
        self.conn.commit()
    
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
        # time.sleep(6)
        self.num_searches += 1
        # 1. Check if the parsed result is already in our fast database
        cursor = self.conn.execute(
            "SELECT is_match, obs, page, result_titles, is_redirect, redirect_entity "
            "FROM wiki_cache WHERE entity = ?", (entity,)
        )
        row = cursor.fetchone()

        if row:
            self.cache_hits += 1
            # Cache Hit: Restore the environment state directly
            is_match, obs, page, result_titles, is_redirect, redirect_entity = row
            
            if is_redirect:
                # If the cache says this was a "may refer to:" redirect, recurse 
                # identically to the original logic
                self.search_step(redirect_entity)
            else:
                self.obs = obs
                if is_match:
                    self.page = page
                    self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
                else:
                    self.result_titles = json.loads(result_titles)
            
            # # Log to txt file just like the original code does
            # with open("wiki.txt", "a", encoding="utf-8") as f:
            #     f.write(f"Search Entity: {entity}\n")
            #     f.write(f"Observation / Wiki Result:\n{self.obs}\n")
            return

        self.cache_misses += 1
        # 2. Cache Miss: Perform the HTTP Request
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        
        old_time = time.time()
        response_text = self.session.get(search_url).text
        self.search_time += time.time() - old_time
        
        
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        
        # Variables to prepare for caching
        is_match = 0
        db_obs = ""
        db_page = None
        db_result_titles = None
        is_redirect = 0
        db_redirect_entity = None

        if result_divs:  # mismatch
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
            db_obs = self.obs
            db_result_titles = json.dumps(self.result_titles)
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                is_redirect = 1
                db_redirect_entity = "[" + entity + "]"
                self.search_step(db_redirect_entity)
                # self.obs is updated inside the recursive call for writing to wiki.txt below
            else:
                is_match = 1
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
                
                db_obs = self.obs
                db_page = self.page

        # 3. Store the fully parsed result in the database
        self.conn.execute('''
            INSERT OR REPLACE INTO wiki_cache 
            (entity, is_match, obs, page, result_titles, is_redirect, redirect_entity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (entity, is_match, db_obs, db_page, db_result_titles, is_redirect, db_redirect_entity))
        self.conn.commit()

        # Log to txt file
        # with open("wiki.txt", "a", encoding="utf-8") as f:
        #     f.write(f"Search Entity: {entity}\n")
        #     f.write(f"Observation / Wiki Result:\n{self.obs}\n")
  
    def step(self, action):
        reward = 0
        done = False
        action = action.strip()
        
        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            self.search_step(entity)
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:  # reset lookup
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
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }