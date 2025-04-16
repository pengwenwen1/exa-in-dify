from collections.abc import Generator
from typing import Any, Dict, List, Optional
import os
import requests
from datetime import datetime
import json

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class ExaSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """
        Execute a search using the Exa Search API.
        
        Args:
            tool_parameters: Dictionary containing:
                - query: The search query string (required)
                - search_type: "neural", "keyword", or "auto" (default: neural)
                - num_results: Maximum number of results to return (default: 10)
                - include_domains: Comma-separated list of domains to include
                - exclude_domains: Comma-separated list of domains to exclude
                - start_published_date: Start date for filtering results (YYYY-MM-DD)
                - end_published_date: End date for filtering results (YYYY-MM-DD)
                - use_autoprompt: Whether to use Exa's query enhancement (default: True)
                - text_contents: Whether to include full text of results (default: False)
                - highlight_results: Whether to highlight relevant snippets (default: False)
                - category: Filter results by category
                - includeText: Text that must be present in results
                - excludeText: Text that must not be present in results
        
        Returns:
            Generator yielding ToolInvokeMessage with search results (both JSON and text formats)
        """
        try:
            # Get API key from runtime credentials
            api_key = self.runtime.credentials["exa_api_key"]
            
            # Required parameter
            query = tool_parameters.get("query")
            if not query:
                raise ValueError("Search query is required")
                
            # Optional parameters with defaults
            search_type = tool_parameters.get("search_type", "neural")
            num_results = int(tool_parameters.get("num_results", 10))
            include_domains = tool_parameters.get("include_domains", "")
            exclude_domains = tool_parameters.get("exclude_domains", "")
            start_published_date = tool_parameters.get("start_published_date", "")
            end_published_date = tool_parameters.get("end_published_date", "")
            use_autoprompt = tool_parameters.get("use_autoprompt", True)
            text_contents = tool_parameters.get("text_contents", False)
            highlight_results = tool_parameters.get("highlight_results", False)
            category = tool_parameters.get("category", None)
            include_text = tool_parameters.get("includeText", None)
            exclude_text = tool_parameters.get("excludeText", None)
            
            # Process domain lists
            include_domains_list = [d.strip() for d in include_domains.split(",")] if include_domains else []
            exclude_domains_list = [d.strip() for d in exclude_domains.split(",")] if exclude_domains else []
            
            # Build request payload
            payload = {
                "query": query,
                "numResults": num_results,
                "useAutoprompt": use_autoprompt
            }
            
            # Handle search type (neural vs keyword vs auto)
            if search_type == "neural":
                payload["type"] = "neural"
            elif search_type == "keyword":
                payload["type"] = "keyword"
            # If auto, don't set type param and let Exa decide
            
            # Add optional parameters if provided
            if include_domains_list:
                payload["includeDomains"] = include_domains_list
            if exclude_domains_list:
                payload["excludeDomains"] = exclude_domains_list
            if start_published_date:
                payload["startPublishedDate"] = start_published_date
            if end_published_date:
                payload["endPublishedDate"] = end_published_date
            if text_contents:
                payload["textContents"] = True
            if highlight_results:
                payload["highlights"] = True
            if category:
                payload["category"] = category
            if include_text:
                payload["includeText"] = [include_text]  # API expects an array
            if exclude_text:
                payload["excludeText"] = [exclude_text]  # API expects an array
                
            # Make API request
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.exa.ai/search",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            # Return original JSON response
            yield self.create_json_message(result_data)
            
            # Format and return text response in Markdown
            markdown_response = self._format_results_as_markdown(result_data, query)
            yield self.create_text_message(markdown_response)
            
        except requests.RequestException as e:
            error_message = f"Error when calling Exa Search API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f" - Status code: {e.response.status_code}"
                if hasattr(e.response, 'text'):
                    error_message += f" - Response: {e.response.text}"
            
            yield self.create_json_message({
                "status": "error",
                "error": error_message
            })
            
            yield self.create_text_message(f"Error: {error_message}")
        except Exception as e:
            error_message = f"Error: {str(e)}"
            
            yield self.create_json_message({
                "status": "error",
                "error": error_message
            })
            
            yield self.create_text_message(f"Error: {error_message}")
    
    def _format_results_as_markdown(self, api_response: Dict, query: str) -> str:
        """Format API response as a readable Markdown string"""
        results = api_response.get("results", [])
        
        markdown = f"## Exa Search Results\n\n"
        markdown += f"**Query:** {query}\n\n"
        markdown += f"**Total Results:** {len(results)}\n\n"
        
        if not results:
            markdown += "No results found.\n"
            return markdown
        
        # Add results
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            domain = result.get("domain", "Unknown source")
            published_date = result.get("publishedDate", "")
            author = result.get("author", "")
            
            markdown += f"### {i}. [{title}]({url})\n\n"
            
            # Add image if available
            if "image" in result and result["image"]:
                image_url = result["image"]
                markdown += f"![Image from {domain}]({image_url})\n\n"
            
            markdown += f"**Source:** {domain}\n"
            
            if published_date:
                markdown += f"**Published:** {published_date}\n"
            
            if author:
                markdown += f"**Author:** {author}\n"
            
            if "score" in result:
                markdown += f"**Relevance Score:** {result['score']:.2f}\n"
            
            markdown += "\n"
            
            # Add highlights if available
            if "highlights" in result and result["highlights"]:
                markdown += "**Highlights:**\n\n"
                for highlight in result["highlights"]:
                    markdown += f"> {highlight}\n"
                markdown += "\n"
            
            # Add text content if available
            if "text" in result and result["text"]:
                text_excerpt = result["text"]
                # Limit to ~500 characters for readability
                if len(text_excerpt) > 500:
                    text_excerpt = text_excerpt[:500] + "..."
                
                markdown += "**Content Excerpt:**\n\n"
                markdown += f"```\n{text_excerpt}\n```\n\n"
            
            markdown += "---\n\n"
        
        return markdown
