import { useContext } from 'react'
import React from 'react'
import {
  CommandBarButton,
  ContextualMenu,
  DefaultButton,
  Dialog,
  DialogFooter,
  DialogType,
  ICommandBarStyles,
  IContextualMenuItem,
  IStackStyles,
  PrimaryButton,
  Spinner,
  SpinnerSize,
  Stack,
  StackItem,
  Text
} from '@fluentui/react'
import { useBoolean } from '@fluentui/react-hooks'

import { ChatHistoryLoadingState, historyDeleteAll } from '../../api'
import { AppStateContext } from '../../state/AppProvider'

import ChatHistoryList from './ChatHistoryList'

import styles from './ChatHistoryPanel.module.css'

interface ChatHistoryPanelProps {}

export enum ChatHistoryPanelTabs {
  History = 'History'
}

const commandBarStyle: ICommandBarStyles = {
  root: {
    padding: '0',
    display: 'flex',
    justifyContent: 'center',
    backgroundColor: 'transparent'
  }
}

const commandBarButtonStyle: Partial<IStackStyles> = { root: { height: '50px' } }

export function ChatHistoryPanel(_props: ChatHistoryPanelProps) {
  const appStateContext = useContext(AppStateContext)
  const [showContextualMenu, setShowContextualMenu] = React.useState(false)
  const [hideClearAllDialog, { toggle: toggleClearAllDialog }] = useBoolean(true)
  const [clearing, setClearing] = React.useState(false)
  const [clearingError, setClearingError] = React.useState(false)

  const clearAllDialogContentProps = {
    type: DialogType.close,
    title: !clearingError ? '모든 채팅 목록을 삭제하시겠습니까?' : 'Error deleting all of chat history',
    closeButtonAriaLabel: 'Close',
    subText: !clearingError
      ? '모든 채팅 목록이 영구적으로 삭제됩니다.'
      : 'Please try again. If the problem persists, please contact the site administrator.'
  }

  const modalProps = {
    titleAriaId: 'labelId',
    subtitleAriaId: 'subTextId',
    isBlocking: true,
    styles: { main: { maxWidth: 450 } }
  }

  const menuItems: IContextualMenuItem[] = [
    //{ key: 'clearAll', text: 'Clear all chat history', iconProps: { iconName: 'Delete' } }
    { key: 'clearAll', text: '모든 채팅 목록 삭제', iconProps: { iconName: 'Delete' } }
  ]

  const handleHistoryClick = () => {
    appStateContext?.dispatch({ type: 'TOGGLE_CHAT_HISTORY' })
  }

  const onShowContextualMenu = React.useCallback((ev: React.MouseEvent<HTMLElement>) => {
    ev.preventDefault() // don't navigate
    setShowContextualMenu(true)
  }, [])

  const onHideContextualMenu = React.useCallback(() => setShowContextualMenu(false), [])

  const onClearAllChatHistory = async () => {
    setClearing(true)
    const response = await historyDeleteAll()
    if (!response.ok) {
      setClearingError(true)
    } else {
      appStateContext?.dispatch({ type: 'DELETE_CHAT_HISTORY' })
      toggleClearAllDialog()
    }
    setClearing(false)
  }

  const onHideClearAllDialog = () => {
    toggleClearAllDialog()
    setTimeout(() => {
      setClearingError(false)
    }, 2000)
  }

  React.useEffect(() => {}, [appStateContext?.state.chatHistory, clearingError])

  return (
    <section className={styles.container} data-is-scrollable aria-label={'chat history panel'}>
      <Stack horizontal horizontalAlign="space-between" verticalAlign="center" wrap aria-label="chat history header">
        <StackItem>
          <Text
            role="heading"
            aria-level={2}
            style={{
              alignSelf: 'center',
              fontWeight: '600',
              fontSize: '18px',
              marginRight: 'auto',
              paddingLeft: '20px'
            }}>
            {/*Chat history*/}
            채팅 목록
          </Text>
        </StackItem>
        <Stack verticalAlign="start">
          <Stack horizontal styles={commandBarButtonStyle}>
            <CommandBarButton
              iconProps={{ iconName: 'More' }}
              title={'모든 채팅 목록 삭제'}
              onClick={onShowContextualMenu}
              aria-label={'clear all chat history'}
              styles={commandBarStyle}
              role="button"
              id="moreButton"
            />
            <ContextualMenu
              items={menuItems}
              hidden={!showContextualMenu}
              target={'#moreButton'}
              onItemClick={toggleClearAllDialog}
              onDismiss={onHideContextualMenu}
            />
            <CommandBarButton
              iconProps={{ iconName: 'Cancel' }}
              title={'Hide'}
              onClick={handleHistoryClick}
              aria-label={'hide button'}
              styles={commandBarStyle}
              role="button"
            />
          </Stack>
        </Stack>
      </Stack>
      <Stack
        aria-label="chat history panel content"
        styles={{
          root: {
            display: 'flex',
            flexGrow: 1,
            flexDirection: 'column',
            paddingTop: '2.5px',
            maxWidth: '100%'
          }
        }}
        style={{
          display: 'flex',
          flexGrow: 1,
          flexDirection: 'column',
          flexWrap: 'wrap',
          padding: '1px'
        }}>
        <Stack className={styles.chatHistoryListContainer}>
          {appStateContext?.state.chatHistoryLoadingState === ChatHistoryLoadingState.Success &&
            appStateContext?.state.isCosmosDBAvailable.cosmosDB && <ChatHistoryList />}
          {appStateContext?.state.chatHistoryLoadingState === ChatHistoryLoadingState.Fail &&
            appStateContext?.state.isCosmosDBAvailable && (
              <>
                <Stack>
                  <Stack horizontalAlign="center" verticalAlign="center" style={{ width: '100%', marginTop: 10 }}>
                    <StackItem>
                      <Text style={{ alignSelf: 'center', fontWeight: '400', fontSize: 16 }}>
                        {appStateContext?.state.isCosmosDBAvailable?.status && (
                          <span>{appStateContext?.state.isCosmosDBAvailable?.status}</span>
                        )}
                        {!appStateContext?.state.isCosmosDBAvailable?.status && <span>Error loading chat history</span>}
                      </Text>
                    </StackItem>
                    <StackItem>
                      <Text style={{ alignSelf: 'center', fontWeight: '400', fontSize: 14 }}>
                        <span>Chat history can't be saved at this time</span>
                      </Text>
                    </StackItem>
                  </Stack>
                </Stack>
              </>
            )}
          {appStateContext?.state.chatHistoryLoadingState === ChatHistoryLoadingState.Loading && (
            <>
              <Stack>
                <Stack
                  horizontal
                  horizontalAlign="center"
                  verticalAlign="center"
                  style={{ width: '100%', marginTop: 10 }}>
                  <StackItem style={{ justifyContent: 'center', alignItems: 'center' }}>
                    <Spinner
                      style={{ alignSelf: 'flex-start', height: '100%', marginRight: '5px' }}
                      size={SpinnerSize.medium}
                    />
                  </StackItem>
                  <StackItem>
                    <Text style={{ alignSelf: 'center', fontWeight: '400', fontSize: 14 }}>
                      <span style={{ whiteSpace: 'pre-wrap' }}>Loading chat history</span>
                    </Text>
                  </StackItem>
                </Stack>
              </Stack>
            </>
          )}
        </Stack>
      </Stack>
      <Dialog
        hidden={hideClearAllDialog}
        onDismiss={clearing ? () => {} : onHideClearAllDialog}
        dialogContentProps={clearAllDialogContentProps}
        modalProps={modalProps}>
        <DialogFooter>
          {!clearingError && <PrimaryButton onClick={onClearAllChatHistory} disabled={clearing} text="Clear All" />}
          <DefaultButton
            onClick={onHideClearAllDialog}
            disabled={clearing}
            text={!clearingError ? 'Cancel' : 'Close'}
          />
        </DialogFooter>
      </Dialog>
    </section>
  )
}
