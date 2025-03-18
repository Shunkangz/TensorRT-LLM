#!/usr/bin/env python
import asyncio
import copy
import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List, Optional, Union

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                CompletionRequest,
                                                CompletionResponse,
                                                DisaggregatedParams,
                                                ErrorResponse)
from tensorrt_llm.version import __version__ as VERSION

logging.basicConfig(level=logging.INFO)

# yapf: enale
TIMEOUT_KEEP_ALIVE = 10  # seconds.

class OpenAIDisaggServer:

    def __init__(self,
                 ctx_servers: List[str] = None,
                 gen_servers: List[str] = None,
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180):
        self.ctx_servers = ctx_servers
        self.gen_servers = gen_servers
        self.ctx_server_idx = 0
        self.gen_server_idx = 0

        if (len(self.gen_servers) == 0):
            raise ValueError("At least one generation server must be provided")

        if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") != "1" and len(ctx_servers) == 0:
            raise ValueError("At least one context server must be provided")

        # Session will be initialized in lifespan
        self.session: Optional[aiohttp.ClientSession] = None

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Create a persistent aiohttp ClientSession
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, keepalive_timeout=300),
                timeout=aiohttp.ClientTimeout(total=req_timeout_secs))

            logging.info("Waiting for context and generation servers to be ready")
            await self.wait_for_servers_ready(server_start_timeout_secs)
            yield
            await self.session.close()  # Ensure session cleanup

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return JSONResponse(status_code=400, content={"error": str(exc)})

        self.register_routes()

    @staticmethod
    def create_error_response(
            message: str,
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        raise HTTPException(status_code=500, detail=f"Internal server error {message}")

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_chat_completion,
                               methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def merge_streaming_responses(self, ctx_response, gen_server, gen_req):
        # First yield the context response if it's not None
        if ctx_response is not None:
            # Remove the disaggregated params from the context response
            data = ctx_response.model_dump()
            del data['choices'][0]['disaggregated_params']
            data = json.dumps(data)
            yield f"data: {data}\n\n".encode('utf-8')

        # Then yield the generation responses
        gen_response = await self.send_request(gen_server, gen_req)
        async for chunk in gen_response.body_iterator:
            yield chunk

    async def openai_completion(self, req: CompletionRequest) -> Response:
        try:
            gen_req = copy.deepcopy(req)
            if not isinstance(req.prompt, str):
                # Check if it's a list and contains integers
                if type(req.prompt) is list and len(req.prompt) == 1:
                    req.prompt = req.prompt[0]
                elif not isinstance(req.prompt, list) or not all(isinstance(x, int) for x in req.prompt):
                    raise ValueError("Disaggregated server currently only supports single string prompt or list of integers in request")

            ctx_response = None
            if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1":
                # Hard-code first token, ctx_request_id for testing
                gen_req.disaggregated_params = DisaggregatedParams(request_type="generation_only", first_gen_tokens=[7], ctx_request_id=1, encoded_opaque_state=None, draft_tokens=None)
                # Since KV cache for prompt tokens will be uninitialized, need to ignore eos
                gen_req.ignore_eos = True
            else:
                # Pick a context server
                ctx_server = self.get_next_server(self.ctx_servers, "context")
                logging.info("Sending request to ctx server: %s", ctx_server)

                # Send request to context server
                req.max_tokens = 1
                req.disaggregated_params = DisaggregatedParams(request_type="context_only")
                # Disable streaming for context server
                req.stream = False
                req.stream_options = None
                ctx_response = await self.send_completion_request(ctx_server, req)

                #TODO: Context server should skip de-tokenization and return raw tokens
                choices = ctx_response.choices
                if len(choices) > 1:
                    raise ValueError("Disagg server returned more than one choice. This is currently not supported in disaggregated server.")
                if choices[0].disaggregated_params is None:
                    raise ValueError("Context server did not return disaggregated params")

                # Append disaggregates parameters to generation request
                gen_req.disaggregated_params = choices[0].disaggregated_params

            gen_req.disaggregated_params.request_type = "generation_only"

            # Pick a generation server and send request
            gen_server = self.get_next_server(self.gen_servers, "generation")
            logging.info("Sending request to gen server: %s", gen_server)
<<<<<<< HEAD
=======
            gen_response = await self.send_completion_request(gen_server, gen_req)
>>>>>>> d45ef81e (Add support of chat completion in PD)

            if not gen_req.stream:
                gen_response = await self.send_request(gen_server, gen_req)
                return gen_response
            else:
                # Return a streaming response that combines both context and generation responses
                return StreamingResponse(
                    self.merge_streaming_responses(ctx_response, gen_server, gen_req),
                    media_type="text/event-stream"
                )

        except CppExecutorError as e:
            # If internal executor error is raised, shutdown the server
            logging.exception(e)
            signal.raise_signal(signal.SIGINT)
        except HTTPException as e:
            raise e  # Re-raise HTTP exceptions properly
        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail=f"Internal server error {str(e)}")

    async def openai_chat_completion(self, req: ChatCompletionRequest) -> Response:

        try:
            gen_req = copy.deepcopy(req)

            # Pick a context server
            ctx_server = self.get_next_server(self.ctx_servers, "context")
            logging.info("Sending request to ctx server: %s", ctx_server)

            # Send request to context server
            req.max_completion_tokens = 1
            req.disaggregated_params = DisaggregatedParams(request_type="context_only")
            # Disable streaming for context server
            req.stream = False
            req.stream_options = None
            ctx_response = await self.send_chat_request(ctx_server, req)

            #TODO: Context server should skip de-tokenization and return raw tokens
            choices = ctx_response.choices
            if len(choices) > 1:
                raise ValueError("Disagg server returned more than one choice. This is currently not supported in disaggregated server.")
            if choices[0].disaggregated_params is None:
                raise ValueError("Context server did not return disaggregated params")

            # Append disaggregates parameters to generation request
            gen_req.disaggregated_params = choices[0].disaggregated_params
            gen_req.disaggregated_params.request_type = "generation_only"

            # Pick a generation server and send request
            gen_server = self.get_next_server(self.gen_servers, "generation")
            logging.info("Sending request to gen server: %s", gen_server)
            gen_response = await self.send_chat_request(gen_server, gen_req)

            return gen_response
        except CppExecutorError as e:
            # If internal executor error is raised, shutdown the server
            logging.exception(e)
            signal.raise_signal(signal.SIGINT)
        except HTTPException as e:
            raise e  # Re-raise HTTP exceptions properly
        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail=f"Internal server error {str(e)}")

    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()

    def get_next_server(self, servers: List[str], server_type: str) -> str:
        """Round-robin selection of next available server"""
        if not servers:
            raise ValueError(f"No {server_type} servers available")

        # Pick context and gen servers in round-robin fashion
        # TODO: In future, use endpoint to monitor load and pick the least loaded server
        if server_type == "context":
            server = servers[self.ctx_server_idx]
            self.ctx_server_idx = (self.ctx_server_idx + 1) % len(servers)
        else:
            server = servers[self.gen_server_idx]
            self.gen_server_idx = (self.gen_server_idx + 1) % len(servers)

        return server


    async def create_completion_generator(self, url: str, request: CompletionRequest):
        async with self.session.post(url + "/v1/completions", json=request.model_dump(exclude_unset=True)) as response:

            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                if not request.stream:
                    raise ValueError("Received an event-stream although request stream was False")

                try:
                    async for line in response.content.iter_any():
                        if line:
                            yield line
                            await asyncio.sleep(0)

                except Exception as e:
                    logging.error(f"Unexpected error in stream: {e}")
                    raise

    async def create_chat_generator(self, url: str, request: ChatCompletionRequest):
        async with self.session.post(url + "/v1/chat/completions", json=request.model_dump(exclude_unset=True)) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                if not request.stream:
                    raise ValueError("Received an event-stream although request stream was False")

                try:
                    async for line in response.content.iter_any():
                        if line:
                            yield line
                            await asyncio.sleep(0)

                except Exception as e:
                    logging.error(f"Unexpected error in stream: {e}")
                    raise

    async def send_completion_request(self, url: str, request: CompletionRequest) -> Union[CompletionResponse, StreamingResponse]:

        if request.stream:
            response_generator = self.create_completion_generator(url, request)
            return StreamingResponse(content=response_generator, media_type="text/event-stream")
        else:
            """Send an asynchronous request and return JSON response"""
            async with self.session.post(url + "/v1/completions", json=request.model_dump(exclude_unset=True)) as response:

                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    raise ValueError("Received an event-stream although request stream was False")

                response_dict = await response.json()
                if not response.ok:
                    logging.error(f"Received failed response {response_dict}")
                    response.raise_for_status()
                return CompletionResponse(**response_dict)

    async def send_chat_request(self, url: str, request: ChatCompletionRequest) -> ChatCompletionResponse:

        if request.stream:
            response_generator = self.create_chat_generator(url, request)
            return StreamingResponse(content=response_generator, media_type="text/event-stream")
        else:
            async with self.session.post(url + "/v1/chat/completions", json=request.model_dump(exclude_unset=True)) as response:

                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    raise ValueError("Received an event-stream although request stream was False")

                response_dict = await response.json()
                response.raise_for_status()
                return ChatCompletionResponse(**response_dict)

    async def check_server_ready(self, server_url: str) -> bool:
        try:
            async with self.session.get(server_url+"/health") as response:
                return response.status == 200
        except Exception:
            return False

    async def wait_for_servers_ready(self, server_start_timeout_secs: int = 180):
        async def are_servers_ready():
            context_ready = all([await self.check_server_ready(url) for url in self.ctx_servers])
            generation_ready = all([await self.check_server_ready(url) for url in self.gen_servers])
            return context_ready and generation_ready

        async def check_all_servers_ready():
            while not await are_servers_ready():
                wait_time = 3
                logging.info("Context and generation servers are not ready. Waiting...")
                await asyncio.sleep(wait_time)

        try:
            await asyncio.wait_for(check_all_servers_ready(), timeout=server_start_timeout_secs)
        except asyncio.CancelledError:
            raise TimeoutError("Timeout waiting for context and generation servers to be ready")
