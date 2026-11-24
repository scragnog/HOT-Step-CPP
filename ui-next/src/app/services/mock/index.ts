// The simulated server: every service of the boundary (services/index.ts), registered at once. main.tsx imports this
// before anything asks for a service; package 10's real adapters take its place there (a file of their own that
// registers the same names), and nothing else changes.

import { defineService } from '../../register/services.ts';
import { mockAdapters } from './adapters.ts';
import { buildSession } from './build.ts';
import { mockGenerate } from './generate.ts';
import { mockLists } from './lists.ts';
import { mockLogs } from './logs.ts';
import { mockModels } from './models.ts';
import { recordings } from './recording.ts';
import { references } from './references.ts';
import { mockScore } from './score.ts';

defineService('lists', mockLists());
defineService('models', mockModels());
defineService('adapters', mockAdapters());
defineService('references', references);
defineService('logs', mockLogs());
defineService('recording', recordings);
defineService('score', mockScore());
defineService('build', buildSession);
/* last: the simulated generation server reads the models service's card */
defineService('generate', mockGenerate());
