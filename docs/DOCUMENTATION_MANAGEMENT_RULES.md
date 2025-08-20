# Documentation Management Rules and Guidelines

## Overview
This document establishes the standardized rules and procedures for managing documentation throughout the Gamma Hedge project lifecycle.

## Directory Structure

```
docs/
├── README.md                    # Main documentation index
├── MASTER_DOCUMENTATION.md     # Unified primary documentation
├── DOCUMENTATION_MANAGEMENT_RULES.md # This file
├── active/                      # Current active documentation
│   ├── current_todo.md         # Current task documentation
│   └── work_in_progress/       # Ongoing task documents
├── templates/                   # Documentation templates
│   ├── task_template.md        # Template for new tasks
│   ├── analysis_template.md    # Template for analysis documents
│   └── progress_template.md    # Template for progress tracking
├── archive/                     # Historical and completed documentation
│   ├── completed_tasks/        # Archived completed task documents
│   ├── legacy/                 # Old system documentation
│   └── external_analysis/      # External analysis reports
└── generated/                   # Auto-generated documentation
    ├── api_docs/               # API documentation
    └── reports/                # Automated reports
```

## Documentation Lifecycle Rules

### 1. Task Documentation Lifecycle

#### Phase 1: Task Planning
1. **Create Task TODO Document**
   - Location: `docs/active/current_todo.md`
   - Template: `docs/templates/task_template.md`
   - Content: Task description, objectives, steps, timeline
   - Update: Real-time as task progresses

#### Phase 2: Task Execution  
1. **Step-by-Step Updates**
   - Update current_todo.md after each step completion
   - Create detailed work documents in `docs/active/work_in_progress/`
   - Document decisions, issues, and solutions

#### Phase 3: Task Completion
1. **Summary and Archive**
   - Create task summary document
   - Archive all task-related documents to `docs/archive/completed_tasks/`
   - Update master documentation with key outcomes
   - Clean up active directory

### 2. Document Categories and Management

#### Active Documents (`docs/active/`)
**Purpose**: Currently being worked on
**Retention**: Until task completion
**Management Rules**:
- Maximum 3-5 active documents at any time
- Must have clear ownership and update schedule
- Regular cleanup every task completion

#### Template Documents (`docs/templates/`)
**Purpose**: Standard formats for new documents
**Retention**: Permanent, version controlled
**Management Rules**:
- Templates updated only through formal review
- Must include all required sections
- Standardized naming and structure

#### Archive Documents (`docs/archive/`)
**Purpose**: Historical reference and completed work
**Retention**: Permanent for reference
**Management Rules**:
- Organized by completion date and task type
- Searchable metadata in document headers
- Regular organization quarterly

#### Generated Documents (`docs/generated/`)
**Purpose**: Auto-generated content (API docs, reports)
**Retention**: Regenerated as needed
**Management Rules**:
- Never manually edited
- Clear generation timestamp and source
- Regular regeneration schedule

### 3. Master Documentation Maintenance

#### MASTER_DOCUMENTATION.md Rules
1. **Content**: Current system state, key decisions, architecture
2. **Updates**: After every major task completion
3. **Structure**: Standardized sections (see template)
4. **Ownership**: Project maintainer responsibility
5. **Review**: All updates require review before merge

#### Integration Process
1. **Source**: Extract key information from completed task docs
2. **Consolidation**: Merge relevant information into master doc
3. **Verification**: Ensure consistency and accuracy
4. **Archive**: Move source documents to archive
5. **Cleanup**: Remove redundant information

### 4. Document Naming Conventions

#### Active Documents
- `current_todo.md` - Current task tracking
- `TASK_[YYYY-MM-DD]_[brief_description].md` - Specific task docs
- `WIP_[component_name]_[description].md` - Work in progress

#### Archive Documents  
- `COMPLETED_[YYYY-MM-DD]_[task_name].md` - Completed tasks
- `LEGACY_[original_name].md` - Archived old documents
- `EXTERNAL_[date]_[source]_[description].md` - External analysis

#### Templates
- `[type]_template.md` - Document templates
- Must include placeholder sections marked with `[PLACEHOLDER]`

### 5. Content Standards

#### Document Headers
All documents must include:
```markdown
# [Document Title]

**Status**: [Active|Completed|Archived|Template]
**Created**: YYYY-MM-DD
**Last Updated**: YYYY-MM-DD
**Owner**: [Primary maintainer]
**Related Tasks**: [Links to related documents]
**Dependencies**: [Prerequisites or dependent documents]

## Summary
[Brief description of document purpose and key content]
```

#### Content Requirements
1. **Clear Purpose**: Every document must state its purpose
2. **Structured Content**: Use consistent heading structure
3. **Actionable Items**: Clear TODO items and next steps
4. **Cross-References**: Links to related documents
5. **Update History**: Track significant changes

### 6. Quality Control

#### Review Process
1. **Self-Review**: Author reviews before archiving
2. **Peer Review**: For master documentation updates
3. **Consistency Check**: Ensure alignment with standards
4. **Completeness Verification**: All required sections present

#### Cleanup Schedule
1. **Weekly**: Review active documents, update status
2. **Task Completion**: Archive completed task documents
3. **Monthly**: Organize archive, update master documentation
4. **Quarterly**: Full documentation system review

### 7. Tool Integration

#### Version Control
- All documentation tracked in git
- Meaningful commit messages for documentation changes
- Branch protection for master documentation

#### Automation
- Auto-generate API documentation
- Automated link checking for internal references
- Template compliance checking

#### Search and Discovery
- Consistent metadata for searchability
- Clear document relationships and dependencies
- Regular index updates

## Implementation Guidelines

### For New Tasks
1. Copy task template to `docs/active/`
2. Fill in task-specific information
3. Update as work progresses
4. Archive upon completion

### For External Documents
1. Evaluate relevance and accuracy
2. Place in appropriate archive subfolder
3. Create evaluation document if needed
4. Reference in master documentation if applicable

### For System Changes
1. Update master documentation immediately
2. Archive obsolete documents
3. Update templates if needed
4. Notify team of changes

## Compliance and Enforcement

### Required Practices
- All new tasks must use standardized templates
- Master documentation must be updated within 24 hours of task completion
- Archive cleanup must occur within 1 week of task completion
- Template updates require formal approval

### Review Checkpoints
- Task completion reviews include documentation check
- Monthly documentation system health checks
- Quarterly comprehensive documentation audits

### Quality Metrics
- Documentation coverage (% of features documented)
- Update frequency and timeliness
- User feedback on documentation quality
- Search and discovery effectiveness

This documentation management system ensures consistent, maintainable, and useful documentation throughout the project lifecycle.