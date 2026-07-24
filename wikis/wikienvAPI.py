import ast
import json
import time
import gymnasium as gym
import requests
import requests_cache
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import logging



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

    #TEST
    # Define o diretório base (padrão local se não existir o /scratch)
    scratch_dir = "/scratch/wikiCache"

    if os.path.exists(scratch_dir):
      db_path = os.path.join(scratch_dir, 'wiki_cache_db')
    else:
      db_path = 'wiki_cache_db'
    self.session = requests.Session()
    # self.session = requests_cache.CachedSession(db_path, expire_after=None, backend='sqlite',use_cache_dir=False,cache_control=False,stale_if_error=True)
    self.session.headers.update({
            'User-Agent': 'WikiRLEnvBot/1.0 (Research/Testing; contact: your_email@example.com)'
        })
    retries = Retry(
        total=5, 
        backoff_factor=1, 
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries,pool_connections=150, pool_maxsize=150)
    self.session.mount('https://', adapter)
    self.session.mount('http://', adapter)
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
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
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret

  def search_step(self, entity):
        api_url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": entity,
            "format": "json"
        }

        old_time = time.time()
        raw_response = self.session.get(api_url, params=params)
        self.search_time += time.time() - old_time
        self.num_searches += 1

        # FIX: Catch non-JSON responses from the initial API call
        try:
            response = raw_response.json()
        except requests.exceptions.JSONDecodeError:
            self.obs = f"API Error: Expected JSON but got status {raw_response.status_code}."
            return

        pages = response.get("query", {}).get("pages", {})
        if not pages:
            self.obs = f"Could not find {entity}. (Invalid API response format)"
            return
            
        page_id = list(pages.keys())[0]

        if page_id == "-1":
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": entity,
                "format": "json"
            }
            raw_search_resp = self.session.get(api_url, params=search_params,timeout=200)
            
            # FIX: Catch non-JSON responses from the fallback search API call
            try:
                search_resp = raw_search_resp.json()
                search_results = search_resp.get("query", {}).get("search", [])
                self.result_titles = [res["title"] for res in search_results]
                self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
            except requests.exceptions.JSONDecodeError:
                self.obs = f"Search API Error: Expected JSON but got status {raw_search_resp.status_code}."
        
        else:
            page_data = pages[page_id]
            text = page_data.get("extract", "")

            if "may refer to:" in text.lower() or "disambiguation" in text.lower():
                self.search_step("[" + entity + "]")
            else:
                self.page = text
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
                
        # with open("wikiAPI.txt", "a", encoding="utf-8") as f:
        #     f.write(f"Search Entity: {entity}\n")
        #     f.write(f"Observation / Wiki Result:\n{self.obs}\n")
  
  def step(self, action):
    reward = 0
    done = False
    action = action.strip()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
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
    }
