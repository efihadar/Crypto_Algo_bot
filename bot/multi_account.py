# bot/multi_account.py
"""
Multi-Account Manager for trading across multiple Bybit accounts.
Handles loading, initializing, and managing multiple trading accounts.
"""
import os
import json
from typing import Any, Dict, List, Optional
from loguru import logger
from pathlib import Path

class MultiAccountManager:
    """
    Manages multiple trading accounts with their own configurations,
    sessions, and trading components.
    """
    def __init__(self):
        """Initialize the multi-account manager."""
        self._accounts: Dict[str, Dict[str, Any]] = {}
        self._active_account_id: Optional[str] = None
        
    def load_accounts(self) -> int:
        """
        Load accounts from environment variables or config file.
        
        Supports two modes:
        1. Single account from BYBIT_API_KEY/BYBIT_API_SECRET env vars
        2. Multiple accounts from ACCOUNTS_JSON env var or accounts.json file
        
        Returns:
            Number of accounts loaded
        """
        accounts_loaded = 0
        
        # Try loading from ACCOUNTS_JSON environment variable
        accounts_json = os.getenv("ACCOUNTS_JSON", "")
        
        if accounts_json:
            try:
                accounts_data = json.loads(accounts_json)
                for acc in accounts_data:
                    acc_id = acc.get("id") or acc.get("name", f"account_{accounts_loaded}")
                    self._accounts[acc_id] = {
                        "id": acc_id,
                        "name": acc.get("name", acc_id),
                        "api_key": acc.get("api_key"),
                        "api_secret": acc.get("api_secret"),
                        "config_file": acc.get("config_file", "config.ini"),
                        "enabled": acc.get("enabled", True),
                        # Components will be added during initialization
                        "session": None,
                        "config": None,
                        "strategy": None,
                        "risk_manager": None,
                        "order_manager": None,
                        "smart_safety": None,
                        "emergency_exit": None,
                    }
                    if self._accounts[acc_id]["enabled"]:
                        accounts_loaded += 1
                        logger.info(f"📂 Loaded account: {acc_id}")
                        
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse ACCOUNTS_JSON: {e}")
        
        # Try loading from accounts.json file
        if accounts_loaded == 0:
            accounts_file = Path("accounts.json")
            if accounts_file.exists():
                try:
                    with open(accounts_file, "r") as f:
                        accounts_data = json.load(f)
                    
                    for acc in accounts_data.get("accounts", accounts_data):
                        if isinstance(acc, dict):
                            acc_id = acc.get("id") or acc.get("name", f"account_{accounts_loaded}")
                            self._accounts[acc_id] = {
                                "id": acc_id,
                                "name": acc.get("name", acc_id),
                                "api_key": acc.get("api_key"),
                                "api_secret": acc.get("api_secret"),
                                "config_file": acc.get("config_file", "config.ini"),
                                "enabled": acc.get("enabled", True),
                                "session": None,
                                "config": None,
                                "strategy": None,
                                "risk_manager": None,
                                "order_manager": None,
                                "smart_safety": None,
                                "emergency_exit": None,
                            }
                            if self._accounts[acc_id]["enabled"]:
                                accounts_loaded += 1
                                logger.info(f"📂 Loaded account from file: {acc_id}")
                                
                except Exception as e:
                    logger.error(f"❌ Failed to load accounts.json: {e}")
        
        # Fallback to single account from environment variables
        if accounts_loaded == 0:
            api_key = os.getenv("BYBIT_API_KEY", "")
            api_secret = os.getenv("BYBIT_API_SECRET", "")
            
            if api_key and api_secret:
                acc_id = "main"
                
                # Support multiple env var naming conventions
                account_name = (
                    os.getenv("ACCOUNT_1_NAME") or
                    os.getenv("ACCOUNT_NAME") or
                    "Main Account"
                )
                config_file = (
                    os.getenv("ACCOUNT_1_CONFIG") or
                    os.getenv("CONFIG_FILE") or
                    "config.ini"
                )
                
                self._accounts[acc_id] = {
                    "id": acc_id,
                    "name": account_name,
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "config_file": config_file,
                    "enabled": True,
                    "session": None,
                    "config": None,
                    "strategy": None,
                    "risk_manager": None,
                    "order_manager": None,
                    "smart_safety": None,
                    "emergency_exit": None,
                }
                accounts_loaded = 1
                logger.info(f"📂 Loaded account: {account_name}")
    
    def get_account_ids(self) -> List[str]:
        """
        Get list of all account IDs.
        Returns:
            List of account ID strings
        """
        return [
            acc_id for acc_id, acc in self._accounts.items() 
            if acc.get("enabled", True)
        ]
    
    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """
        Get account data by ID.
        Args:
            account_id: The account identifier   
        Returns:
            Account dictionary or None if not found
        """
        return self._accounts.get(account_id)
    
    def get_all_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all enabled accounts.
        Returns:
            List of account dictionaries
        """
        return [
            acc for acc in self._accounts.values() 
            if acc.get("enabled", True)
        ]
    
    def get_active_account(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently active account.
        Returns:
            Active account dictionary or None
        """
        if self._active_account_id:
            return self._accounts.get(self._active_account_id)
        return None
    
    def set_active_account(self, account_id: str) -> bool:
        """
        Set the active account by ID.
        Args:
            account_id: The account identifier to activate
        Returns:
            True if successful, False if account not found
        """
        if account_id in self._accounts:
            self._active_account_id = account_id
            logger.info(f"🔄 Switched to account: {account_id}")
            return True
        logger.warning(f"⚠️ Account not found: {account_id}")
        return False
    
    def initialize_account_components(
        self,
        account_id: str,
        session: Any = None,
        config: Any = None,
        risk_manager: Any = None,
        order_manager: Any = None,
        strategy: Any = None,
        smart_safety: Any = None,
        emergency_exit: Any = None,
    ) -> bool:
        """
        Initialize trading components for an account.
        Args:
            account_id: The account identifier
            session: BybitSession instance
            config: ConfigLoader instance
            risk_manager: RiskManager instance
            order_manager: OrderManager instance
            strategy: BybitStrategy instance
            smart_safety: SmartSafetyManager instance
            emergency_exit: EmergencyExitManager instance
        Returns:
            True if successful, False if account not found
        """
        if account_id not in self._accounts:
            logger.error(f"❌ Cannot initialize unknown account: {account_id}")
            return False
        
        acc = self._accounts[account_id]
        
        if session is not None:
            acc["session"] = session
        if config is not None:
            acc["config"] = config
        if risk_manager is not None:
            acc["risk_manager"] = risk_manager
        if order_manager is not None:
            acc["order_manager"] = order_manager
        if strategy is not None:
            acc["strategy"] = strategy
        if smart_safety is not None:
            acc["smart_safety"] = smart_safety
        if emergency_exit is not None:
            acc["emergency_exit"] = emergency_exit
            
        logger.debug(f"✅ Components initialized for account: {account_id}")
        return True
    
    def get_component(self, account_id: str, component_name: str) -> Optional[Any]:
        """
        Get a specific component from an account.
        Args:
            account_id: The account identifier
            component_name: Name of the component (session, config, etc.)
        Returns:
            The component or None if not found
        """
        acc = self._accounts.get(account_id)
        if acc:
            return acc.get(component_name)
        return None
    
    def get_session(self, account_id: str) -> Optional[Any]:
        """Get session for an account."""
        return self.get_component(account_id, "session")
    
    def get_strategy(self, account_id: str) -> Optional[Any]:
        """Get strategy for an account."""
        return self.get_component(account_id, "strategy")
    
    def get_order_manager(self, account_id: str) -> Optional[Any]:
        """Get order manager for an account."""
        return self.get_component(account_id, "order_manager")
    
    def get_risk_manager(self, account_id: str) -> Optional[Any]:
        """Get risk manager for an account."""
        return self.get_component(account_id, "risk_manager")
    
    def disable_account(self, account_id: str) -> bool:
        """
        Disable an account from trading.
        
        Args:
            account_id: The account identifier
            
        Returns:
            True if successful, False if account not found
        """
        if account_id in self._accounts:
            self._accounts[account_id]["enabled"] = False
            logger.warning(f"🚫 Account disabled: {account_id}")
            return True
        return False
    
    def enable_account(self, account_id: str) -> bool:
        """
        Enable an account for trading.
        
        Args:
            account_id: The account identifier
            
        Returns:
            True if successful, False if account not found
        """
        if account_id in self._accounts:
            self._accounts[account_id]["enabled"] = True
            logger.info(f"✅ Account enabled: {account_id}")
            return True
        return False
    
    def get_total_balance(self) -> float:
        """
        Get combined balance across all enabled accounts.
        
        Returns:
            Total balance in USDT
        """
        total = 0.0
        for acc in self.get_all_accounts():
            session = acc.get("session")
            if session:
                try:
                    info = session.get_account_info()
                    if info:
                        total += float(info.get("balance", 0) or 0)
                except Exception as e:
                    logger.debug(f"⚠️ Failed to get balance for {acc['id']}: {e}")
        return total
    
    def get_total_equity(self) -> float:
        """
        Get combined equity across all enabled accounts.
        
        Returns:
            Total equity in USDT
        """
        total = 0.0
        for acc in self.get_all_accounts():
            session = acc.get("session")
            if session:
                try:
                    info = session.get_account_info()
                    if info:
                        total += float(info.get("equity", 0) or 0)
                except Exception as e:
                    logger.debug(f"⚠️ Failed to get equity for {acc['id']}: {e}")
        return total
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all accounts.
        
        Returns:
            Dictionary with account summary
        """
        accounts_summary = []
        total_balance = 0.0
        total_equity = 0.0
        total_positions = 0
        
        for acc in self.get_all_accounts():
            acc_data = {
                "id": acc["id"],
                "name": acc["name"],
                "enabled": acc["enabled"],
                "balance": 0.0,
                "equity": 0.0,
                "positions": 0,
            }
            
            session = acc.get("session")
            if session:
                try:
                    info = session.get_account_info()
                    if info:
                        acc_data["balance"] = float(info.get("balance", 0) or 0)
                        acc_data["equity"] = float(info.get("equity", 0) or 0)
                        total_balance += acc_data["balance"]
                        total_equity += acc_data["equity"]
                except Exception:
                    pass
            
            order_manager = acc.get("order_manager")
            if order_manager and hasattr(order_manager, "active_orders"):
                acc_data["positions"] = len(order_manager.active_orders)
                total_positions += acc_data["positions"]
            
            accounts_summary.append(acc_data)
        
        return {
            "total_accounts": len(self._accounts),
            "enabled_accounts": len(self.get_all_accounts()),
            "total_balance": total_balance,
            "total_equity": total_equity,
            "total_positions": total_positions,
            "accounts": accounts_summary,
        }

# Singleton instance
_multi_account_manager: Optional[MultiAccountManager] = None

def get_multi_account_manager() -> MultiAccountManager:
    """
    Get the singleton MultiAccountManager instance.
    Returns:
        MultiAccountManager instance
    """
    global _multi_account_manager
    
    if _multi_account_manager is None:
        _multi_account_manager = MultiAccountManager()
    
    return _multi_account_manager